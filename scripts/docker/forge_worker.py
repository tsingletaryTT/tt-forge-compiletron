#!/usr/bin/env python3
"""
Worker process for compiling models on a single TT chip.
Runs inside Docker container with TT_VISIBLE_DEVICES set.
"""
import os
import sys
import time
from datetime import datetime

def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    chip_id = os.environ.get("TT_VISIBLE_DEVICES", "?")
    print(f"[{timestamp}] [Chip {chip_id}] {msg}", flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: forge_worker.py <test_name>")
        sys.exit(1)

    test_name = sys.argv[1]
    chip_id = os.environ.get("TT_VISIBLE_DEVICES", "unknown")

    log("=" * 60)
    log(f"Starting worker for chip {chip_id}")
    log(f"Test: {test_name}")
    log("=" * 60)

    try:
        # Import forge (will be isolated to this chip via TT_VISIBLE_DEVICES)
        log("Importing forge...")
        import forge
        import torch

        log(f"✓ Forge version: {forge.__version__ if hasattr(forge, '__version__') else 'unknown'}")
        log(f"✓ Torch version: {torch.__version__}")

        # Create simple test model
        log("Creating test model...")

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 128)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        inputs = torch.randn(1, 128)

        log("✓ Model created")

        # Compile for this chip
        log(f"Compiling model on chip {chip_id}...")
        start_time = time.time()

        compiled = forge.compile(
            model,
            sample_inputs=[inputs],
            module_name=f"{test_name}_chip_{chip_id}",
        )

        elapsed = time.time() - start_time
        log(f"✓ Compilation completed in {elapsed:.2f}s")

        log("=" * 60)
        log(f"Worker for chip {chip_id} finished successfully!")
        log("=" * 60)

        return 0

    except Exception as e:
        log(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
