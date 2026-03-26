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
# Uses pane IDs (%N) captured with -P at creation time.
# Pane IDs are globally unique and unaffected by base-index / pane-base-index.
#
# Split order (bottom strip first = full-width status bar):
#   1. new-session           → P_TL  (top-left, full window)
#   2. split-window -v 15%   → P_STA (bottom strip, full width)   split from P_TL
#   3. split-window -h 50%   → P_TR  (top-right)                  split from P_TL
#   4. split-window -v 50%   → P_BL  (bottom-left)                split from P_TL
#   5. split-window -v 50%   → P_BR  (bottom-right)               split from P_TR

tmux new-session -d -s "$SESSION"

# Capture initial pane ID (top-left)
P_TL=$(tmux display-message -t "$SESSION" -p "#{pane_id}")

# Step 2: full-width status bar (split from top-left pane, so it spans full width)
P_STA=$(tmux split-window -v -p 15 -t "$P_TL" -P -F "#{pane_id}")

# Step 3: top-right (split top-left horizontally)
P_TR=$(tmux split-window -h -p 50 -t "$P_TL" -P -F "#{pane_id}")

# Step 4: bottom-left (split top-left vertically)
P_BL=$(tmux split-window -v -p 50 -t "$P_TL" -P -F "#{pane_id}")

# Step 5: bottom-right (split top-right vertically)
P_BR=$(tmux split-window -v -p 50 -t "$P_TR" -P -F "#{pane_id}")

# ── Pane titles ───────────────────────────────────────────────────────────────

tmux select-pane -t "$P_TL"  -T "  Chip 0  "
tmux select-pane -t "$P_TR"  -T "  Chip 2  "
tmux select-pane -t "$P_BL"  -T "  Chip 1  "
tmux select-pane -t "$P_BR"  -T "  Chip 3  "
tmux select-pane -t "$P_STA" -T "  Status  "

tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "
tmux set -t "$SESSION" pane-border-style "fg=colour240"
tmux set -t "$SESSION" pane-active-border-style "fg=colour214,bold"

# ── Launch Docker containers ──────────────────────────────────────────────────

tmux send-keys -t "$P_TL"  "$(docker_cmd 0)" C-m
tmux send-keys -t "$P_TR"  "$(docker_cmd 2)" C-m
tmux send-keys -t "$P_BL"  "$(docker_cmd 1)" C-m
tmux send-keys -t "$P_BR"  "$(docker_cmd 3)" C-m

# ── Status pane: live docker container watch ──────────────────────────────────

tmux send-keys -t "$P_STA" \
    "watch -n2 'echo \"Running containers:\"; docker ps --filter name=forge_chip --format \"  {{.Names}}  {{.Status}}\" 2>/dev/null || echo \"  (none)\"; echo \"\"; echo \"Done: \$(docker ps -a --filter name=forge_chip --filter status=exited --format x 2>/dev/null | wc -l)/4 chips\"'" \
    C-m

# Focus top-left
tmux select-pane -t "$P_TL"

# ── Attach ────────────────────────────────────────────────────────────────────

echo ""
echo "  ┌──────────────┬──────────────┐"
echo "  │  Chip 0      │  Chip 2      │"
echo "  ├──────────────┼──────────────┤"
echo "  │  Chip 1      │  Chip 3      │"
echo "  ├──────────────┴──────────────┤"
echo "  │  Status (docker watch)      │"
echo "  └─────────────────────────────┘"
echo ""
echo "  Ctrl+B + arrow keys = navigate panes"
echo "  Ctrl+B + D          = detach"
echo "  tmux attach -t $SESSION  = reattach"
echo ""

tmux attach-session -t "$SESSION"
