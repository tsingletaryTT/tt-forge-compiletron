#!/bin/bash
# 4-Way TT-Forge Demo with tmux visualization
# Shows 4 chips compiling models simultaneously in a 2x2 grid

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  TT-Forge 4-Way tmux Demo Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker image exists
if ! docker image inspect tt-forge-compiletron:minimal &>/dev/null; then
    echo -e "${RED}✗ Docker image 'tt-forge-compiletron:minimal' not found${NC}"
    echo -e "  Build it with: docker build -f Dockerfile.minimal -t tt-forge-compiletron:minimal ."
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker image available"

# Check tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}✗ tmux not installed${NC}"
    echo -e "  Install with: sudo apt install tmux"
    exit 1
fi

echo -e "${GREEN}✓${NC} tmux available"
echo ""

# Kill existing session if it exists
tmux kill-session -t forge_demo 2>/dev/null || true

# Test name
TEST_NAME="${1:-demo_test}"

echo -e "${BLUE}Creating 4-pane tmux session...${NC}"

# Create tmux session with 4 panes (2x2 grid)
tmux new-session -d -s forge_demo -n "TT-Forge 4-Way Demo"

# Split into 4 panes
tmux split-window -h -t forge_demo
tmux split-window -v -t forge_demo:0.0
tmux split-window -v -t forge_demo:0.1

# Set up each pane with a Docker command
for chip_id in {0..3}; do
    echo -e "${BLUE}  Setting up pane for chip ${chip_id}...${NC}"

    # Set pane title
    tmux send-keys -t forge_demo:0.${chip_id} "echo '=== TT Chip ${chip_id} ===' && echo ''" C-m

    # Prepare Docker command (don't run yet - user will press Enter)
    tmux send-keys -t forge_demo:0.${chip_id} \
        "docker run --rm --name forge_chip_${chip_id} --device=/dev/tenstorrent:/dev/tenstorrent -e TT_VISIBLE_DEVICES=${chip_id} -e TT_METAL_ARCH_NAME=blackhole -e TT_MESH_GRAPH_DESC_PATH=/root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/forge/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto --entrypoint python3 tt-forge-compiletron:minimal /app/scripts/forge_worker.py ${TEST_NAME}"
done

# Make panes even
tmux select-layout -t forge_demo tiled

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Attach to tmux: ${BLUE}tmux attach -t forge_demo${NC}"
echo -e "  2. In each pane, press ${BLUE}Enter${NC} to start compilation"
echo -e "  3. Watch all 4 chips compile simultaneously!"
echo ""
echo -e "${YELLOW}tmux shortcuts:${NC}"
echo -e "  • Switch panes: ${BLUE}Ctrl+B then arrow keys${NC}"
echo -e "  • Detach: ${BLUE}Ctrl+B then D${NC}"
echo -e "  • Kill session: ${BLUE}tmux kill-session -t forge_demo${NC}"
echo ""
echo -e "${YELLOW}Recording tips:${NC}"
echo -e "  • Maximize terminal before recording"
echo -e "  • Increase font size (Ctrl+Shift++)"
echo -e "  • Use: OBS Studio, SimpleScreenRecorder, or asciinema"
echo ""
