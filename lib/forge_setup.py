#!/usr/bin/env python3
"""
Forge installation and environment validation helper.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def check_forge_installed() -> bool:
    """
    Check if forge is importable.

    Returns:
        True if forge can be imported, False otherwise
    """
    try:
        # Try importing with tt-forge-fe in path
        forge_fe_path = Path.home() / "tt-forge-fe"
        if forge_fe_path.exists():
            sys.path.insert(0, str(forge_fe_path))

        import forge
        return True
    except ImportError:
        return False


def get_forge_path() -> Optional[Path]:
    """
    Find tt-forge-fe installation.

    Returns:
        Path to tt-forge-fe if found, None otherwise
    """
    # Check common locations
    locations = [
        Path.home() / "tt-forge-fe",
        Path("/opt/tt-forge-fe"),
        Path("/usr/local/tt-forge-fe"),
    ]

    for path in locations:
        if path.exists() and (path / "forge").exists():
            return path

    return None


def check_forge_environment() -> Dict[str, any]:
    """
    Check Forge environment activation status.

    Returns:
        Dict with environment status:
            - forge_installed: bool
            - forge_path: Path or None
            - env_activated: bool
            - env_vars: dict of relevant env vars
    """
    forge_path = get_forge_path()
    forge_installed = forge_path is not None

    # Check if environment is activated
    env_vars = {
        'TTFORGE_TOOLCHAIN_DIR': os.environ.get('TTFORGE_TOOLCHAIN_DIR'),
        'TTFORGE_VENV_DIR': os.environ.get('TTFORGE_VENV_DIR'),
        'TTMLIR_TOOLCHAIN_DIR': os.environ.get('TTMLIR_TOOLCHAIN_DIR'),
        'ARCH_NAME': os.environ.get('ARCH_NAME'),
    }

    env_activated = any(v is not None for v in env_vars.values())

    return {
        'forge_installed': forge_installed,
        'forge_path': forge_path,
        'env_activated': env_activated,
        'env_vars': env_vars,
    }


def check_dependencies() -> Dict[str, any]:
    """
    Verify required dependencies.

    Returns:
        Dict with dependency status:
            - python_version: str (e.g., '3.12.1')
            - python_ok: bool (True if >= 3.12)
            - tt_metal: Path or None
            - tt_metal_ok: bool
            - pytorch_installed: bool
            - torchvision_installed: bool
    """
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info >= (3, 12)

    # Check tt-metal
    tt_metal_home = os.environ.get('TT_METAL_HOME')
    if not tt_metal_home:
        tt_metal_home = str(Path.home() / "tt-metal")

    tt_metal_path = Path(tt_metal_home)
    tt_metal_ok = tt_metal_path.exists() and (tt_metal_path / "build_Release").exists()

    # Check PyTorch
    try:
        import torch
        pytorch_installed = True
    except ImportError:
        pytorch_installed = False

    try:
        import torchvision
        torchvision_installed = True
    except ImportError:
        torchvision_installed = False

    return {
        'python_version': python_version,
        'python_ok': python_ok,
        'tt_metal': tt_metal_path if tt_metal_ok else None,
        'tt_metal_ok': tt_metal_ok,
        'pytorch_installed': pytorch_installed,
        'torchvision_installed': torchvision_installed,
    }


def install_forge(dest: Optional[Path] = None, branch: str = "main") -> Tuple[bool, str]:
    """
    Clone and build tt-forge-fe (takes 45-60 minutes).

    Args:
        dest: Destination directory (default: ~/tt-forge-fe)
        branch: Git branch to checkout (default: main)

    Returns:
        (success, message) tuple
    """
    if dest is None:
        dest = Path.home() / "tt-forge-fe"

    if dest.exists():
        return False, f"Directory already exists: {dest}"

    print(f"Cloning tt-forge-fe to {dest}...")
    print("This will take a few minutes...")

    try:
        # Clone repository
        result = subprocess.run(
            ["git", "clone", "https://github.com/tenstorrent/tt-forge-fe.git", str(dest)],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            return False, f"Clone failed: {result.stderr}"

        # Checkout branch
        if branch != "main":
            subprocess.run(
                ["git", "checkout", branch],
                cwd=dest,
                capture_output=True,
                text=True
            )

        print(f"\nBuilding tt-forge-fe...")
        print("This will take 45-60 minutes...")
        print("You can monitor progress in another terminal with:")
        print(f"  tail -f {dest}/build.log\n")

        # Build (this takes a long time)
        build_script = dest / "build_forge.sh"
        if not build_script.exists():
            return False, "build_forge.sh not found"

        # Run build in background, save output to log
        with open(dest / "build.log", "w") as log:
            result = subprocess.run(
                ["bash", str(build_script)],
                cwd=dest,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=4000  # 1 hour timeout
            )

        if result.returncode != 0:
            return False, f"Build failed. Check {dest}/build.log for details"

        return True, f"Successfully installed tt-forge-fe to {dest}"

    except subprocess.TimeoutExpired:
        return False, "Build timed out (> 1 hour)"
    except Exception as e:
        return False, f"Installation failed: {e}"


def print_environment_status():
    """Print comprehensive environment status."""
    print("TT-Forge Environment Status")
    print("=" * 60)

    # Check Forge
    forge_env = check_forge_environment()
    if forge_env['forge_installed']:
        print(f"✓ Forge installed: {forge_env['forge_path']}")
    else:
        print(f"✗ Forge not found")

    if forge_env['env_activated']:
        print(f"✓ Forge environment activated")
        for k, v in forge_env['env_vars'].items():
            if v:
                print(f"    {k}={v}")
    else:
        print(f"⚠️  Forge environment not activated")
        if forge_env['forge_path']:
            print(f"    Run: source {forge_env['forge_path']}/env/activate")

    # Check dependencies
    deps = check_dependencies()
    print(f"\nDependencies:")
    print(f"  Python: {deps['python_version']} {'✓' if deps['python_ok'] else '✗ (need >= 3.12)'}")
    print(f"  tt-metal: {'✓' if deps['tt_metal_ok'] else '✗'} {deps['tt_metal'] if deps['tt_metal_ok'] else 'not found'}")
    print(f"  PyTorch: {'✓' if deps['pytorch_installed'] else '✗'}")
    print(f"  torchvision: {'✓' if deps['torchvision_installed'] else '✗'}")


def get_activation_instructions() -> str:
    """
    Get instructions for activating Forge environment.

    Returns:
        String with activation instructions
    """
    forge_path = get_forge_path()
    if not forge_path:
        return "Forge not installed. Run: compiletron setup --install-forge"

    activate_script = forge_path / "env" / "activate"
    if activate_script.exists():
        return f"source {activate_script}"
    else:
        return f"Forge installed but env/activate not found. May need to rebuild."


# Example usage
if __name__ == '__main__':
    print_environment_status()

    print("\n" + "=" * 60)
    print("Activation instructions:")
    print(get_activation_instructions())
