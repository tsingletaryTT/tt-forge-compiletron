#!/bin/bash
# Launch 4 Docker containers simultaneously, one per chip
# Each container is isolated to a single chip via TT_VISIBLE_DEVICES

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  TT-Forge 4-Way Parallel Compilation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker image exists
if ! docker image inspect tt-forge-compiletron:minimal &>/dev/null; then
    echo -e "${RED}✗ Docker image 'tt-forge-compiletron:minimal' not found${NC}"
    echo -e "  Build it with: docker build -f Dockerfile.minimal -t tt-forge-compiletron:minimal ."
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker image found: tt-forge-compiletron:minimal"

# Check if chips are available
if ! ls /dev/tenstorrent* &>/dev/null; then
    echo -e "${RED}✗ No Tenstorrent devices found at /dev/tenstorrent*${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Tenstorrent devices available"
echo ""

# Create log directory
LOG_DIR="/tmp/forge_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}Logs will be saved to: ${LOG_DIR}${NC}"
echo ""

# Test name
TEST_NAME="${1:-parallel_test}"

echo -e "${BLUE}Launching 4 Docker containers (one per chip)...${NC}"
echo ""

# Launch all 4 containers in parallel (background)
for chip_id in {0..3}; do
    echo -e "${BLUE}  Starting chip ${chip_id}...${NC}"

    docker run --rm \
        --name "forge_chip_${chip_id}" \
        --device=/dev/tenstorrent:/dev/tenstorrent \
        -e TT_VISIBLE_DEVICES="${chip_id}" \
        -e TT_METAL_ARCH_NAME="blackhole" \
        -e TT_MESH_GRAPH_DESC_PATH="/root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/forge/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto" \
        --entrypoint python3 \
        tt-forge-compiletron:minimal \
        /app/scripts/forge_worker.py "${TEST_NAME}" \
        > "${LOG_DIR}/chip${chip_id}.log" 2>&1 &

    # Store PID
    eval "PID_${chip_id}=$!"
done

echo ""
echo -e "${GREEN}✓${NC} All 4 containers launched"
echo ""
echo -e "${YELLOW}Monitoring progress...${NC}"
echo -e "  (Tail logs with: tail -f ${LOG_DIR}/chip*.log)"
echo ""

# Wait for all containers and track their status
success_count=0
fail_count=0

for chip_id in {0..3}; do
    eval "pid=\$PID_${chip_id}"

    echo -e "${BLUE}Waiting for chip ${chip_id} (PID ${pid})...${NC}"

    if wait "$pid"; then
        echo -e "${GREEN}✓ Chip ${chip_id} completed successfully${NC}"
        ((success_count++))
    else
        exit_code=$?
        echo -e "${RED}✗ Chip ${chip_id} failed (exit code ${exit_code})${NC}"
        ((fail_count++))
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  ${GREEN}Success:${NC} ${success_count}/4 chips"
echo -e "  ${RED}Failed:${NC}  ${fail_count}/4 chips"
echo ""
echo -e "  Logs: ${LOG_DIR}/"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✓ All chips completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some chips failed. Check logs for details.${NC}"
    exit 1
fi
