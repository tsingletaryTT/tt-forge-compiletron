#!/bin/bash
# 4-Way TT-Forge compilation with deterministic tmux 2x2 grid + status bar
#
# Layout:
#   ┌──────────────┬──────────────┐
#   │  Chip 0      │  Chip 2      │  (live docker output)
#   ├──────────────┼──────────────┤
#   │  Chip 1      │  Chip 3      │  (live docker output)
#   ├──────────────┴──────────────┤
#   │  Status / docker ps watch   │  (full-width)
#   └─────────────────────────────┘
#
# Usage:
#   bash scripts/docker/run_4way_tmux.sh
#   bash scripts/docker/run_4way_tmux.sh demo_test   # custom test name

set -e

SESSION="forge-parallel"
IMAGE="tt-forge-compiletron:full"
MESH_DESC_PATH="/app/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
TEST_NAME="${1:-parallel_test}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "✗ Docker image '$IMAGE' not found"
    echo "  Build it with: docker build -t $IMAGE ."
    echo "  Note: First build takes 2-3 hours"
    exit 1
fi

if ! ls /dev/tenstorrent* &>/dev/null; then
    echo "✗ No Tenstorrent devices found at /dev/tenstorrent*"
    exit 1
fi

if ! command -v tmux &>/dev/null; then
    echo "✗ tmux not installed: sudo apt install tmux"
    exit 1
fi

# Kill any leftover containers from a previous run
for chip_id in 0 1 2 3; do
    docker rm -f "forge_chip_${chip_id}" &>/dev/null || true
done

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Docker command builder ────────────────────────────────────────────────────

docker_cmd() {
    local chip_id=$1
    echo "docker run --rm \
        --name forge_chip_${chip_id} \
        --device=/dev/tenstorrent:/dev/tenstorrent \
        --shm-size=16g \
        -e TT_VISIBLE_DEVICES=${chip_id} \
        -e TT_METAL_ARCH_NAME=blackhole \
        -e TT_MESH_GRAPH_DESC_PATH=${MESH_DESC_PATH} \
        ${IMAGE} \
        python3 /app/scripts/docker/forge_worker.py ${TEST_NAME}; \
        echo ''; echo '════ CHIP ${chip_id} DONE ════'; read -p 'Press Enter to close...'"
}

# ── Build deterministic tmux layout ──────────────────────────────────────────
#
# Creation order is the key to determinism:
#   1. new-session          → pane 0 (full window)
#   2. split-window -v 15%  → pane 1 (bottom 15%, full width) — status bar
#   3. split-window -h 50%  → pane 2 (top-right)              on pane 0
#   4. split-window -v 50%  → pane 3 (bottom-left)            on pane 0
#   5. split-window -v 50%  → pane 4 (bottom-right)           on pane 2
#
# Result:
#   pane 0 = top-left    (Chip 0)
#   pane 2 = top-right   (Chip 2)
#   pane 3 = bottom-left (Chip 1)
#   pane 4 = bottom-right(Chip 3)
#   pane 1 = bottom strip (Status)

tmux new-session -d -s "$SESSION"

# Create bottom status strip first — this way it spans the full width
tmux split-window -v -p 15 -t "$SESSION:0.0"

# Split top area left|right
tmux select-pane -t "$SESSION:0.0"
tmux split-window -h -p 50 -t "$SESSION:0.0"

# Split top-left into top/bottom
tmux select-pane -t "$SESSION:0.0"
tmux split-window -v -p 50 -t "$SESSION:0.0"

# Split top-right into top/bottom
tmux select-pane -t "$SESSION:0.2"
tmux split-window -v -p 50 -t "$SESSION:0.2"

# ── Assign pane titles ────────────────────────────────────────────────────────

tmux select-pane -t "$SESSION:0.0" -T "Chip 0"
tmux select-pane -t "$SESSION:0.2" -T "Chip 2"
tmux select-pane -t "$SESSION:0.3" -T "Chip 1"
tmux select-pane -t "$SESSION:0.4" -T "Chip 3"
tmux select-pane -t "$SESSION:0.1" -T "Status"

# Enable pane titles in status bar
tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "
tmux set -t "$SESSION" pane-border-style "fg=colour240"
tmux set -t "$SESSION" pane-active-border-style "fg=colour250,bold"

# ── Launch Docker containers in each chip pane ────────────────────────────────

tmux send-keys -t "$SESSION:0.0" "$(docker_cmd 0)" C-m
tmux send-keys -t "$SESSION:0.2" "$(docker_cmd 2)" C-m
tmux send-keys -t "$SESSION:0.3" "$(docker_cmd 1)" C-m
tmux send-keys -t "$SESSION:0.4" "$(docker_cmd 3)" C-m

# ── Status pane: watch which containers are still running ─────────────────────

tmux send-keys -t "$SESSION:0.1" \
    "watch -n2 'echo \"forge containers:\"; docker ps --filter name=forge_chip --format \"  {{.Names}}  status={{.Status}}\" 2>/dev/null || echo \"  (none running)\"; echo \"\"; echo \"Chips done: \$(docker ps -a --filter name=forge_chip --filter status=exited --format \"{{.Names}}\" 2>/dev/null | wc -l)/4\"'" \
    C-m

# Focus top-left chip pane
tmux select-pane -t "$SESSION:0.0"

# ── Attach ────────────────────────────────────────────────────────────────────

echo ""
echo "  Layout:"
echo "  ┌──────────────┬──────────────┐"
echo "  │  Chip 0      │  Chip 2      │"
echo "  ├──────────────┼──────────────┤"
echo "  │  Chip 1      │  Chip 3      │"
echo "  ├──────────────┴──────────────┤"
echo "  │  Status (docker watch)      │"
echo "  └─────────────────────────────┘"
echo ""
echo "  Ctrl+B + arrow = navigate panes"
echo "  Ctrl+B + D     = detach"
echo "  tmux attach -t $SESSION  = reattach"
echo ""

tmux attach-session -t "$SESSION"
