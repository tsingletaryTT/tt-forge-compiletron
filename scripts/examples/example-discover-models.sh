#!/bin/bash
# Example: Discover New Models from Multiple Sources
#
# This script demonstrates how to discover models automatically from:
# 1. TT-Forge test repositories
# 2. HuggingFace model hub (by family)
#
# Usage:
#   ./scripts/examples/example-discover-models.sh [family]
#
# Examples:
#   ./scripts/examples/example-discover-models.sh resnet
#   ./scripts/examples/example-discover-models.sh bert
#   ./scripts/examples/example-discover-models.sh vit

set -e

FAMILY="${1:-resnet}"

echo "======================================================================"
echo "  Model Discovery Example"
echo "======================================================================"
echo ""
echo "Discovering ${FAMILY} models from multiple sources..."
echo ""

# Create output directory
mkdir -p discovery_results

echo "1. Searching HuggingFace model hub..."
echo "----------------------------------------------------------------------"
python3 compiletron.py discover huggingface \
    --family "${FAMILY}" \
    --limit 20 \
    --verbose \
    --save "discovery_results/huggingface_${FAMILY}.json"

echo ""
echo ""
echo "2. Scanning TT-Forge repositories (if available)..."
echo "----------------------------------------------------------------------"

# Check if tt-forge-fe exists
if [ -d ~/tt-forge-fe ]; then
    python3 compiletron.py discover forge \
        --forge-path ~/tt-forge-fe \
        --verbose \
        --save "discovery_results/forge_models.json"
else
    echo "⚠️  tt-forge-fe not found at ~/tt-forge-fe"
    echo "   Skipping Forge repository scan"
    echo "   Install with: git clone https://github.com/tenstorrent/tt-forge-fe.git ~/tt-forge-fe"
fi

echo ""
echo ""
echo "======================================================================"
echo "  Discovery Complete!"
echo "======================================================================"
echo ""
echo "Results saved to: discovery_results/"
echo ""
echo "Next steps:"
echo "  1. Review discovered models: cat discovery_results/huggingface_${FAMILY}.json"
echo "  2. Test compilation: python3 compiletron.py discover test discovery_results/huggingface_${FAMILY}.json"
echo "  3. Add successful models to lib/models.py"
echo ""
echo "Try other families:"
echo "  ./scripts/examples/example-discover-models.sh bert"
echo "  ./scripts/examples/example-discover-models.sh efficientnet"
echo "  ./scripts/examples/example-discover-models.sh vit"
echo ""
