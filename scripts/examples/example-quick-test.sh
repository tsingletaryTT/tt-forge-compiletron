#!/bin/bash
# Example: Quick Test - Run 5 fast models on single chip
#
# This demonstrates the simplest workflow for validating your Forge setup.
# Uses the 5 fastest models (< 5s each) on chip 0.
#
# Expected time: ~15-20 seconds total
# Hardware required: Any Tenstorrent device (1+ chips)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example: Quick Test (5 fastest models)"
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

# Run quick test
echo "Running quick test on chip 0..."
echo

python3 compiletron.py run --quick --chip 0

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Quick test complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Next steps:"
echo "  • View results: python3 compiletron.py results"
echo "  • Try parallel: ./scripts/examples/example-parallel-sweep.sh"
echo "  • Explore models: python3 compiletron.py models families"
echo
