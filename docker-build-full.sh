#!/bin/bash
# Build full self-contained Docker image with tt-metal and tt-forge-fe built from source
#
# WARNING: This is a LONG build:
# - Build time: 2-3 hours
# - RAM required: 32GB+
# - Disk space: 50GB+ for build, 30GB final image
# - Internet: Will download several GB of dependencies
#
# Arguments:
#   --tt-metal-commit COMMIT   Specific tt-metal commit (default: main)
#   --tt-forge-commit COMMIT   Specific tt-forge-fe commit (default: main)
#   --no-cache                 Force rebuild without cache
#   --tag TAG                  Additional tag for the image

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Default values
TT_METAL_COMMIT=${TT_METAL_COMMIT:-main}
TT_FORGE_COMMIT=${TT_FORGE_COMMIT:-main}
USE_CACHE=true
EXTRA_TAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tt-metal-commit)
            TT_METAL_COMMIT="$2"
            shift 2
            ;;
        --tt-forge-commit)
            TT_FORGE_COMMIT="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --tag)
            EXTRA_TAG="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display build information
echo ""
info "Building TT-Forge Compiletron (Full Self-Contained Image)"
echo "========================================================"
echo ""
warning "This build will take 2-3 hours!"
echo ""
echo "Configuration:"
echo "  TT-Metal commit: $TT_METAL_COMMIT"
echo "  TT-Forge commit: $TT_FORGE_COMMIT"
echo "  Use cache: $USE_CACHE"
echo "  Extra tag: ${EXTRA_TAG:-none}"
echo ""
echo "Requirements:"
echo "  • 32GB+ RAM"
echo "  • 50GB+ disk space"
echo "  • Good internet connection"
echo "  • 2-3 hours of time"
echo ""

# Ask for confirmation
read -p "Continue with build? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Build cancelled"
    exit 0
fi

# Build command
BUILD_CMD="docker build -f Dockerfile.full-build"

# Add build args
BUILD_CMD="$BUILD_CMD --build-arg TT_METAL_COMMIT=$TT_METAL_COMMIT"
BUILD_CMD="$BUILD_CMD --build-arg TT_FORGE_COMMIT=$TT_FORGE_COMMIT"

# Cache option
if [ "$USE_CACHE" = false ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

# Tags
BUILD_CMD="$BUILD_CMD --tag tt-forge-compiletron:full-latest"
if [ -n "$EXTRA_TAG" ]; then
    BUILD_CMD="$BUILD_CMD --tag tt-forge-compiletron:$EXTRA_TAG"
fi

# Progress
BUILD_CMD="$BUILD_CMD --progress=plain"

# Context
BUILD_CMD="$BUILD_CMD ."

# Start build
info "Starting build at $(date)"
echo ""
info "Build command: $BUILD_CMD"
echo ""

# Run build
if eval "$BUILD_CMD"; then
    echo ""
    success "Build complete!"
    echo ""
    echo "Image tags:"
    echo "  • tt-forge-compiletron:full-latest"
    if [ -n "$EXTRA_TAG" ]; then
        echo "  • tt-forge-compiletron:$EXTRA_TAG"
    fi
    echo ""
    echo "Image size:"
    docker images tt-forge-compiletron:full-latest --format "  • {{.Size}}"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: docker run --rm tt-forge-compiletron:full-latest test"
    echo "  2. Detect hardware: docker run --rm --device=/dev/tenstorrent tt-forge-compiletron:full-latest detect"
    echo "  3. Compile models: docker run --rm --device=/dev/tenstorrent tt-forge-compiletron:full-latest compile --quick"
    echo ""
else
    echo ""
    error "Build failed!"
    exit 1
fi
