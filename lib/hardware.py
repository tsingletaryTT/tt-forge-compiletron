#!/usr/bin/env python3
"""
Hardware detection and configuration for Tenstorrent devices.
Supports 1 to 32+ chips with automatic detection and configuration.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def detect_hardware() -> Dict[str, any]:
    """
    Detect Tenstorrent hardware configuration.

    Returns:
        Dict with keys:
            - num_chips: Number of detected chips (int)
            - board_type: Board type (str, e.g., 'P300C', 'P150')
            - arch: Architecture (str, e.g., 'blackhole', 'wormhole_b0')
            - devices: List of device info dicts

    Works for N=1 to N=32+ chips.
    """
    try:
        # Run tt-smi with JSON output
        result = subprocess.run(
            ['tt-smi', '-s'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return {
                'num_chips': 0,
                'board_type': 'unknown',
                'arch': 'unknown',
                'devices': [],
                'error': 'tt-smi command failed'
            }

        # Parse JSON output
        data = json.loads(result.stdout)
        # tt-smi uses 'device_info' not 'devices'
        devices = data.get('device_info', [])
        num_chips = len(devices)

        if num_chips == 0:
            return {
                'num_chips': 0,
                'board_type': 'unknown',
                'arch': 'unknown',
                'devices': [],
                'error': 'No devices detected'
            }

        # Get board type from first device
        # Structure: device_info[0].board_info.board_type
        first_device = devices[0]
        board_info = first_device.get('board_info', {})
        board_type = board_info.get('board_type', 'unknown').upper()  # Normalize to uppercase

        # Map board type to architecture
        # tt-smi doesn't include arch directly, so we infer from board type
        board_to_arch = {
            'P300C': 'blackhole',
            'P150': 'blackhole',
            'N300': 'wormhole_b0',
            'P100': 'grayskull',
        }
        arch = board_to_arch.get(board_type, 'unknown')

        # Extract simplified device info for easy access
        simplified_devices = []
        for i, dev in enumerate(devices):
            board_info = dev.get('board_info', {})
            simplified_devices.append({
                'device_index': i,
                'board_type': board_type,
                'bus_id': board_info.get('bus_id', 'unknown'),
                'raw': dev  # Keep full device data for advanced use
            })

        return {
            'num_chips': num_chips,
            'board_type': board_type,
            'arch': arch,
            'devices': simplified_devices
        }

    except FileNotFoundError:
        return {
            'num_chips': 0,
            'board_type': 'unknown',
            'arch': 'unknown',
            'devices': [],
            'error': 'tt-smi not found (not installed or not in PATH)'
        }
    except subprocess.TimeoutExpired:
        return {
            'num_chips': 0,
            'board_type': 'unknown',
            'arch': 'unknown',
            'devices': [],
            'error': 'tt-smi timeout'
        }
    except json.JSONDecodeError as e:
        return {
            'num_chips': 0,
            'board_type': 'unknown',
            'arch': 'unknown',
            'devices': [],
            'error': f'Failed to parse tt-smi output: {e}'
        }
    except Exception as e:
        return {
            'num_chips': 0,
            'board_type': 'unknown',
            'arch': 'unknown',
            'devices': [],
            'error': f'Unexpected error: {e}'
        }


def system_ram_gb() -> float:
    """Return total system RAM in gigabytes."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def safe_max_params_b(num_chips: int) -> float:
    """Compute a safe per-model parameter cap based on available RAM and chip count.

    forge needs roughly 5× model params in RAM at fp32 (weights + compilation IR
    + intermediate tensors). With N chips compiling in parallel each subprocess
    can independently hit that ceiling, so we divide by num_chips.

    Uses 70% of total RAM to leave headroom for the OS, the TUI, and the
    Tenstorrent driver. Returns 0.0 if RAM cannot be determined (disables cap).
    """
    ram = system_ram_gb()
    if ram <= 0 or num_chips <= 0:
        return 0.0
    safe = (ram * 0.70) / (num_chips * 5.0)
    # Round down to a clean breakpoint so the displayed value is readable.
    breakpoints = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 13.0, 20.0, 30.0, 70.0]
    for bp in reversed(breakpoints):
        if safe >= bp:
            return bp
    return max(1.0, round(safe, 1))


def get_chip_config(chip_id: int, num_chips: int, arch: str, board_type: str) -> Dict[str, str]:
    """
    Generate environment variables for isolating one chip.

    Args:
        chip_id: 0-based chip index
        num_chips: Total chips available (for validation)
        arch: Architecture name (e.g., 'blackhole')
        board_type: Board type (e.g., 'P300C')

    Returns:
        Dict of environment variables to set for this chip

    Raises:
        ValueError: If chip_id is invalid
    """
    if chip_id < 0 or chip_id >= num_chips:
        raise ValueError(f"chip_id {chip_id} out of range (0-{num_chips-1})")

    env = {
        'TT_VISIBLE_DEVICES': str(chip_id),
        'TT_METAL_ARCH_NAME': arch,
    }

    # Add mesh descriptor path for single-chip isolation (if needed for board type)
    if board_type in ['P300C', 'P150']:
        # P300C and P150 require mesh descriptor for single-chip operation
        mesh_desc_path = _find_mesh_descriptor(arch)
        if mesh_desc_path:
            env['TT_MESH_GRAPH_DESC_PATH'] = str(mesh_desc_path)

    return env


def calculate_model_distribution(total_models: int, num_chips: int) -> List[Tuple[int, List[int]]]:
    """
    Calculate round-robin distribution of models across chips.

    Args:
        total_models: Total number of models to distribute
        num_chips: Number of chips available

    Returns:
        List of (chip_id, model_indices) tuples

    Example with 108 models, 4 chips:
        Chip 0: [0,4,8,12,16,...]  (27 models)
        Chip 1: [1,5,9,13,17,...]  (27 models)
        Chip 2: [2,6,10,14,18,...] (27 models)
        Chip 3: [3,7,11,15,19,...] (27 models)

    Example with 108 models, 8 chips:
        Chip 0: [0,8,16,24,...]    (14 models)
        Chip 1: [1,9,17,25,...]    (14 models)
        ... and so on
    """
    if num_chips <= 0:
        raise ValueError("num_chips must be positive")
    if total_models <= 0:
        raise ValueError("total_models must be positive")

    distribution = []
    for chip_id in range(num_chips):
        # Round-robin: chip N gets models N, N+K, N+2K, ...
        model_indices = list(range(chip_id, total_models, num_chips))
        distribution.append((chip_id, model_indices))

    return distribution


def validate_mesh_descriptor(board_type: str, arch: str) -> Optional[Path]:
    """
    Check if mesh descriptor file exists for single-chip isolation.

    Args:
        board_type: Board type (e.g., 'P300C')
        arch: Architecture (e.g., 'blackhole')

    Returns:
        Path to mesh descriptor if found, None otherwise

    Note:
        Required for P300C boards, may not be needed for others.
        The descriptor tells UMD how to handle single-chip isolation.
    """
    if board_type not in ['P300C', 'P150']:
        # Not needed for these boards
        return None

    return _find_mesh_descriptor(arch)


def _find_mesh_descriptor(arch: str) -> Optional[Path]:
    """
    Find mesh descriptor file for given architecture.

    Args:
        arch: Architecture name

    Returns:
        Path to descriptor file if found, None otherwise
    """
    # Common locations for mesh descriptor files
    tt_metal_home = Path(os.environ.get('TT_METAL_HOME', Path.home() / 'tt-metal'))

    descriptor_paths = [
        # Release build
        tt_metal_home / 'build_Release/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto',
        # Debug build
        tt_metal_home / 'build_Debug/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto',
        # Alternative location
        tt_metal_home / 'tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto',
    ]

    for path in descriptor_paths:
        if path.exists():
            return path

    return None


def print_hardware_info(hw_info: Dict) -> None:
    """
    Print hardware information in a nice format.

    Args:
        hw_info: Hardware info dict from detect_hardware()
    """
    if 'error' in hw_info:
        print(f"❌ Error: {hw_info['error']}")
        return

    print(f"✓ Detected {hw_info['num_chips']} Tenstorrent chip(s)")
    print(f"  Board type: {hw_info['board_type']}")
    print(f"  Architecture: {hw_info['arch']}")

    if hw_info['devices']:
        print(f"\n  Devices:")
        for dev in hw_info['devices']:
            dev_id = dev.get('device_index', '?')
            bus_id = dev.get('bus_id', '?')
            print(f"    • Device {dev_id}: {hw_info['board_type']} at {bus_id}")


def get_hardware_summary(hw_info: Dict) -> str:
    """
    Get one-line hardware summary.

    Args:
        hw_info: Hardware info dict

    Returns:
        String like "4x P300C (Blackhole)"
    """
    if 'error' in hw_info:
        return "No hardware detected"

    return f"{hw_info['num_chips']}x {hw_info['board_type']} ({hw_info['arch'].title()})"


# Example usage and testing
if __name__ == '__main__':
    print("TT-Forge Compiletron - Hardware Detection")
    print("=" * 60)

    # Detect hardware
    hw = detect_hardware()
    print_hardware_info(hw)

    if hw['num_chips'] > 0:
        print(f"\n{get_hardware_summary(hw)}")

        # Show round-robin distribution for 108 models
        print(f"\nRound-robin distribution (108 models across {hw['num_chips']} chips):")
        dist = calculate_model_distribution(108, hw['num_chips'])
        for chip_id, model_ids in dist:
            print(f"  Chip {chip_id}: {len(model_ids)} models - indices {model_ids[:5]}...{model_ids[-3:]}")

        # Show chip configuration
        print(f"\nEnvironment configuration for Chip 0:")
        cfg = get_chip_config(0, hw['num_chips'], hw['arch'], hw['board_type'])
        for k, v in cfg.items():
            print(f"  {k}={v}")

        # Validate mesh descriptor
        mesh_path = validate_mesh_descriptor(hw['board_type'], hw['arch'])
        if mesh_path:
            print(f"\n✓ Mesh descriptor found: {mesh_path}")
        else:
            if hw['board_type'] in ['P300C', 'P150']:
                print(f"\n⚠️  Mesh descriptor not found (may be required for {hw['board_type']})")
            else:
                print(f"\n  (Mesh descriptor not needed for {hw['board_type']})")
