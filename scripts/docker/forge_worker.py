#!/usr/bin/env python3
"""
Docker entry-point for single-chip Forge compilation.

Invoked by run_4way_tmux.sh (docker mode) as:
  python3 /app/scripts/docker/forge_worker.py <test_name>

TT_VISIBLE_DEVICES is already set in the environment by docker run -e.

Delegates to lib/worker.py which contains the full visual pipeline:
  - pyfiglet ASCII art banners with rotating Tenstorrent brand colors
  - [1/3][2/3][3/3] progress steps
  - 3-second celebration pause between models
  - Round-robin model distribution: chip N gets models N, N+4, N+8, ...
  - Per-model colored checklist (✓/✗) on completion
  - Stays alive so the tmux pane keeps showing results
"""
import os
import sys
from pathlib import Path

# /app is the project root inside the Docker image
APP_DIR = Path(__file__).resolve().parent.parent.parent  # /app
sys.path.insert(0, str(APP_DIR))

from lib.worker import run_worker
from lib.models import MODEL_LIST


def main():
    # test_name arg is accepted for compatibility but not used by lib/worker
    # (it's a label that was meaningful in the old SimpleModel stub)
    chip_id = int(os.environ.get('TT_VISIBLE_DEVICES', '0'))

    # Round-robin: chip N compiles models N, N+4, N+8, ...
    stride = 4
    model_indices = list(range(chip_id, len(MODEL_LIST), stride))

    results_file = Path(f'/tmp/forge_results_chip_{chip_id}.csv')

    sys.exit(run_worker(chip_id, model_indices, results_file))


if __name__ == '__main__':
    main()
