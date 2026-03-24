#!/bin/bash
# Multi-chip parallel orchestrator
# Dynamically supports 1 to 32+ chips

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 TT-Forge Compiletron - Parallel Orchestrator"
echo "=" >&2

# Detect hardware
NUM_CHIPS=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/lib')
from hardware import detect_hardware
hw = detect_hardware()
print(hw.get('num_chips', 0))
" 2>/dev/null)

if [ "$NUM_CHIPS" -eq 0 ]; then
    echo "❌ No Tenstorrent chips detected"
    echo "   Run 'tt-smi' to check hardware"
    exit 1
fi

echo "Detected: $NUM_CHIPS chip(s)"

# Get architecture
ARCH=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/lib')
from hardware import detect_hardware
hw = detect_hardware()
print(hw.get('arch', 'blackhole'))
" 2>/dev/null)

echo "Architecture: $ARCH"

# Calculate model distribution
echo ""
echo "Calculating round-robin distribution..."

python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/lib')
from hardware import calculate_model_distribution
dist = calculate_model_distribution(101, $NUM_CHIPS)
for chip_id, model_ids in dist:
    print(f'Chip {chip_id}: {len(model_ids)} models')
"

echo ""
echo "⚠️  Full parallel execution not yet implemented."
echo ""
echo "To run parallel compilation:"
echo "  1. Activate Forge: source ~/tt-forge-fe/env/activate"
echo "  2. Use tmux to split terminals"
echo "  3. In each terminal, set TT_VISIBLE_DEVICES=<chip_id>"
echo "  4. Run: python3 $PROJECT_DIR/lib/worker.py"
echo ""
echo "Or use the original implementation:"
echo "  cd ~/tt-forge-creative-demos"
echo "  ./run_parallel_forge.sh"
