#!/bin/bash
# Build TT-Forge Compiletron Docker image

set -e

echo "🐳 Building TT-Forge Compiletron Docker image..."
echo "================================================="
echo ""

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
    --tag tt-forge-compiletron:latest \
    --tag tt-forge-compiletron:$(date +%Y%m%d) \
    --progress=plain \
    .

echo ""
echo "✅ Build complete!"
echo ""
echo "Image tags:"
echo "  • tt-forge-compiletron:latest"
echo "  • tt-forge-compiletron:$(date +%Y%m%d)"
echo ""
echo "Next steps:"
echo "  1. Run tests: ./docker-run.sh test"
echo "  2. Detect hardware: ./docker-run.sh detect"
echo "  3. See all options: ./docker-run.sh help"
