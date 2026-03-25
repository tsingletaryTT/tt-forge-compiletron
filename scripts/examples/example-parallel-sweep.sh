#!/bin/bash
# Example: Parallel Sweep - Run 100 models across all chips
#
# This demonstrates parallel multi-chip execution with round-robin distribution.
# Compiles 100 models distributed evenly across all available chips.
#
# Expected time: ~30 minutes (4 chips), ~60 minutes (2 chips)
# Hardware required: 2+ Tenstorrent devices for best results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example: Parallel Sweep (100 models)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check if Forge environment is activated
if [ -z "$TTFORGE_TOOLCHAIN_DIR" ] && [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
    echo "⚠️  Forge environment not activated!"
    echo
    echo "Please activate Forge first:"
    echo "  source ~/tt-forge-fe/env/activate"
    echo
    exit 1
fi

echo "✓ Forge environment detected"
echo

# Detect hardware
echo "Detecting hardware..."
python3 compiletron.py detect
echo

# Show what will be compiled
echo "Estimating compilation time..."
python3 compiletron.py models estimate --count 100
echo

# Ask for confirmation
read -p "Continue with parallel compilation? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo
echo "Starting parallel compilation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Run parallel orchestrator
bash scripts/run_parallel.sh

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Parallel sweep complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "View results:"
echo "  • Summary: python3 compiletron.py results"
echo "  • Detailed: python3 compiletron.py results view -v"
echo "  • Report: python3 compiletron.py results report --output results_100.md"
echo
