#!/bin/bash
# Example: Family Compilation - Compile all models in a family
#
# This demonstrates how to compile all variants of a model family (e.g., all ResNets).
# Useful for systematic testing of specific architectures.
#
# Expected time: Varies by family (e.g., ResNet ~2 min, EfficientNet ~15 min)
# Hardware required: Any Tenstorrent device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# Default family
FAMILY="${1:-resnet}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Example: Family Compilation ($FAMILY)"
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

# Show available families
echo "Available model families:"
python3 compiletron.py models families
echo

# Show models in selected family
echo "Models in $FAMILY family:"
python3 compiletron.py models list --family $FAMILY
echo

# Estimate time
echo "Estimating compilation time..."
python3 compiletron.py models estimate --family $FAMILY
echo

# Ask for confirmation
read -p "Compile all $FAMILY models? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo
echo "Starting compilation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Run compilation
python3 compiletron.py run --family $FAMILY --chip 0

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Family compilation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Usage:"
echo "  ./example-family-compilation.sh [family]"
echo
echo "Examples:"
echo "  ./example-family-compilation.sh resnet"
echo "  ./example-family-compilation.sh efficientnet"
echo "  ./example-family-compilation.sh vgg"
echo
