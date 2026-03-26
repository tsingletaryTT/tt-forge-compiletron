#!/usr/bin/env python3
"""
Worker process for single-chip Forge compilation (native).
This script runs INSIDE the child process spawned by orchestrator.
"""
import os
import sys
import time
from datetime import datetime

def redirect_output_to_log(log_file):
    """Redirect stdout/stderr to log file"""
    log_fd = open(log_file, 'w', buffering=1)
    sys.stdout = log_fd
    sys.stderr = log_fd
    os.environ['PYTHONUNBUFFERED'] = '1'

def log(chip_id, msg):
    """Log with timestamp and chip ID"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [Chip {chip_id}] {msg}", flush=True)

def main():
    if len(sys.argv) != 3:
        print("Usage: worker_native.py <chip_id> <log_file>")
        sys.exit(1)

    chip_id = int(sys.argv[1])
    log_file = sys.argv[2]

    # Redirect output BEFORE importing anything
    redirect_output_to_log(log_file)

    log(chip_id, "=" * 60)
    log(chip_id, f"Starting worker for chip {chip_id}")
    log(chip_id, f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')}")
    log(chip_id, f"TT_METAL_ARCH_NAME={os.environ.get('TT_METAL_ARCH_NAME')}")
    log(chip_id, f"TT_MESH_GRAPH_DESC_PATH={os.environ.get('TT_MESH_GRAPH_DESC_PATH', 'not set')}")

    # Suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')

    # NOW import forge (after env is set and logs redirected)
    try:
        log(chip_id, "Importing forge...")
        import forge
        import torch
        log(chip_id, f"✓ Forge imported successfully")
        log(chip_id, f"  Forge version: {forge.__version__ if hasattr(forge, '__version__') else 'unknown'}")
    except ImportError as e:
        log(chip_id, f"✗ ERROR: Failed to import forge: {e}")
        sys.exit(1)

    # Create a simple test model
    log(chip_id, "Creating test model...")

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    inputs = torch.randn(1, 128)

    log(chip_id, "Compiling model...")
    try:
        start_time = time.time()
        compiled = forge.compile(
            model,
            sample_inputs=[inputs],
            module_name=f"test_native_chip_{chip_id}",
        )
        compile_time = time.time() - start_time
        log(chip_id, f"✓ Compilation completed in {compile_time:.2f}s")
        log(chip_id, "=" * 60)
        sys.exit(0)
    except Exception as e:
        log(chip_id, f"✗ ERROR during compilation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
