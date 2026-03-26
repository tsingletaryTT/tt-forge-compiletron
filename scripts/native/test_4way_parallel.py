#!/usr/bin/env python3
"""
Test 4-way parallel compilation on native host (no Docker).
Based on tt-forge-creative-demos/parallel_forge_orchestrator.py
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

LOG_DIR = Path("/tmp/forge_parallel_native")

def setup_log_dir():
    """Create clean log directory"""
    LOG_DIR.mkdir(exist_ok=True)
    # Clear old logs
    for log_file in LOG_DIR.glob("chip*.log"):
        log_file.unlink()
    # Create empty log files
    for i in range(4):
        (LOG_DIR / f"chip{i}.log").touch()

def launch_child_for_chip(chip_id):
    """Spawn child process for one chip"""
    env = os.environ.copy()

    # CRITICAL: Pin to single chip
    env["TT_VISIBLE_DEVICES"] = str(chip_id)
    env["TT_METAL_ARCH_NAME"] = "blackhole"

    # CRITICAL: Mesh graph descriptor for single-chip isolation
    # Use standalone tt-metal build (STRICT policy) - matches working tt-forge-creative-demos
    # This is different from the forge-bundled version (RELAXED policy)
    mesh_desc = Path.home() / "tt-metal/build_Release/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
    if mesh_desc.exists():
        env["TT_MESH_GRAPH_DESC_PATH"] = str(mesh_desc)
    else:
        print(f"[Orchestrator] WARNING: Mesh descriptor not found at {mesh_desc}")
        print(f"[Orchestrator] Chip {chip_id} may fail with fabric mesh graph error")

    # Clear conflicting vars (Forge should be isolated)
    env.pop("TT_METAL_HOME", None)
    env.pop("TT_METAL_VERSION", None)
    env.pop("TT_METAL_DEVICE_ID", None)  # Use TT_VISIBLE_DEVICES instead

    # Child script and args
    script_dir = Path(__file__).parent
    worker_script = script_dir / "worker_native.py"

    cmd = [
        "python3",
        str(worker_script),
        str(chip_id),
        str(LOG_DIR / f"chip{chip_id}.log")
    ]

    print(f"[Orchestrator] Launching child for chip {chip_id}")

    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"[Orchestrator] Chip {chip_id} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Orchestrator] Chip {chip_id} failed with exit code {e.returncode}")
        return False

def main():
    print("=" * 80)
    print("TT-FORGE 4-WAY PARALLEL TEST (NATIVE)")
    print(f"Started: {datetime.now()}")
    print("Hardware: 4x Blackhole chips")
    print("=" * 80)

    setup_log_dir()

    print("\nLaunching 4 child processes (one per chip)...")
    print(f"Logs: {LOG_DIR}/chip{{0,1,2,3}}.log")
    print()

    # Launch all 4 children in parallel using thread pool
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(launch_child_for_chip, chip_id)
            for chip_id in range(4)
        ]
        # Wait for all to complete
        results = [future.result() for future in futures]

    successes = sum(results)
    failures = 4 - successes

    print()
    print("=" * 80)
    print(f"RESULTS: {successes}/4 chips succeeded, {failures}/4 failed")
    print(f"Finished: {datetime.now()}")
    print("=" * 80)

    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
