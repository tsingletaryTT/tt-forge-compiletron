#!/bin/bash
# Convenience wrapper for running TT-Forge Compiletron in Docker

set -e

# Default paths (override with environment variables)
TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tt-metal}"
FORGE_HOME="${FORGE_HOME:-$HOME/tt-forge-fe}"

# Colors
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if image exists
if ! docker image inspect tt-forge-compiletron:latest >/dev/null 2>&1; then
    echo "❌ Docker image not found. Building..."
    ./docker-build.sh
fi

# Construct docker run command
DOCKER_CMD="docker run --rm"

# Add device access if available
if [ -e /dev/tenstorrent ]; then
    DOCKER_CMD="$DOCKER_CMD --device=/dev/tenstorrent"
fi

# Add volume mounts if paths exist
if [ -d "$TT_METAL_HOME" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $TT_METAL_HOME:/tt-metal:ro"
else
    info "tt-metal not found at $TT_METAL_HOME (skipping mount)"
fi

if [ -d "$FORGE_HOME" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $FORGE_HOME:/tt-forge-fe:ro"
else
    info "tt-forge-fe not found at $FORGE_HOME (skipping mount)"
fi

# Mount Forge toolchain if available (contains venv with dependencies)
if [ -d "/opt/ttforge-toolchain" ]; then
    DOCKER_CMD="$DOCKER_CMD -v /opt/ttforge-toolchain:/opt/ttforge-toolchain:ro"
fi

# Add persistent volumes
DOCKER_CMD="$DOCKER_CMD -v compiletron-cache:/cache"
DOCKER_CMD="$DOCKER_CMD -v compiletron-results:/results"

# Add interactive flags if running shell
if [ "$1" = "shell" ] || [ "$1" = "bash" ]; then
    DOCKER_CMD="$DOCKER_CMD -it"
fi

# Add image and command
DOCKER_CMD="$DOCKER_CMD tt-forge-compiletron:latest"

# Run with all arguments
$DOCKER_CMD "$@"
