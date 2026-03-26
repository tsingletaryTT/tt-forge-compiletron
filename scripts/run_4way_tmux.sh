#!/bin/bash
# 4-Way TT-Forge compilation viewer: deterministic tmux 2x2 grid + status bar
#
# Works in both Docker and native (direct Forge) modes.
#
# Layout:
#   ┌──────────────┬──────────────┐
#   │  Chip 0      │  Chip 2      │  (live output)
#   ├──────────────┼──────────────┤
#   │  Chip 1      │  Chip 3      │  (live output)
#   ├──────────────┴──────────────┤
#   │  Status                     │  (full-width)
#   └─────────────────────────────┘
#
# Usage:
#   bash scripts/run_4way_tmux.sh                        # auto-detect mode
#   bash scripts/run_4way_tmux.sh --mode native          # native (Forge env)
#   bash scripts/run_4way_tmux.sh --mode docker          # Docker containers
#   bash scripts/run_4way_tmux.sh --mode native --count 5
#   bash scripts/run_4way_tmux.sh --name my_run

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="forge-parallel"
DOCKER_IMAGE="tt-forge-compiletron:full"
MESH_DESC_PATH="/app/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"

# Defaults
MODE=""       # auto-detect if empty
COUNT=10
TEST_NAME="parallel_test"

# ── Parse args ────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)  MODE="$2";      shift 2 ;;
        --count) COUNT="$2";     shift 2 ;;
        --name)  TEST_NAME="$2"; shift 2 ;;
        -*)      echo "Unknown option: $1"; exit 1 ;;
        *)       TEST_NAME="$1"; shift ;;
    esac
done

# ── Auto-detect mode ──────────────────────────────────────────────────────────

if [[ -z "$MODE" ]]; then
    if docker image inspect "$DOCKER_IMAGE" &>/dev/null 2>&1; then
        MODE="docker"
    else
        MODE="native"
    fi
    echo "Auto-detected mode: $MODE"
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if ! command -v tmux &>/dev/null; then
    echo "✗ tmux not installed: sudo apt install tmux"
    exit 1
fi

if [[ "$MODE" == "docker" ]]; then
    if ! docker image inspect "$DOCKER_IMAGE" &>/dev/null; then
        echo "✗ Docker image '$DOCKER_IMAGE' not found"
        echo "  Build it with: docker build -t $DOCKER_IMAGE ."
        echo "  Or use --mode native to run without Docker"
        exit 1
    fi
    if ! ls /dev/tenstorrent* &>/dev/null; then
        echo "✗ No Tenstorrent devices found at /dev/tenstorrent*"
        exit 1
    fi
    for chip_id in 0 1 2 3; do
        docker rm -f "forge_chip_${chip_id}" &>/dev/null || true
    done
elif [[ "$MODE" == "native" ]]; then
    if [[ ! -f "$PROJECT_DIR/compiletron.py" ]]; then
        echo "✗ compiletron.py not found in $PROJECT_DIR"
        exit 1
    fi
    # Kill any stale forge/compiletron processes that might hold device locks.
    # UMD chip locks are not released when processes are killed abruptly.
    if pkill -0 -f "compiletron.py run" 2>/dev/null; then
        echo "Killing stale compiletron processes and waiting for device locks to clear..."
        pkill -f "compiletron.py run" 2>/dev/null || true
        sleep 2
    fi
fi

# ── Build per-chip command ────────────────────────────────────────────────────

chip_cmd() {
    local chip_id=$1
    local done_msg="════ CHIP ${chip_id} DONE ════"
    if [[ "$MODE" == "docker" ]]; then
        echo "docker run --rm \
            --name forge_chip_${chip_id} \
            --device=/dev/tenstorrent:/dev/tenstorrent \
            --shm-size=16g \
            -e TT_VISIBLE_DEVICES=${chip_id} \
            -e TT_METAL_ARCH_NAME=blackhole \
            -e TT_MESH_GRAPH_DESC_PATH=${MESH_DESC_PATH} \
            ${DOCKER_IMAGE} \
            python3 /app/scripts/docker/forge_worker.py ${TEST_NAME}; \
            echo ''; echo '${done_msg}'; read -p 'Press Enter to close...'"
    else
        # Native: activate forge env, then run compiletron directly.
        # TT_MESH_GRAPH_DESC_PATH is required for CUSTOM cluster type (P300 single-chip).
        #
        # Staggered startup: each chip waits chip_id * 4 seconds before initializing
        # UMD. Without staggering all 4 chips race to open the same PCIe device and
        # the losers fail immediately with a lock error.
        local native_mesh="${PROJECT_DIR}/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
        local stagger=$((chip_id * 4))
        local stagger_msg=""
        if [[ $stagger -gt 0 ]]; then
            stagger_msg="echo '[Chip ${chip_id}] Staggered start: waiting ${stagger}s for earlier chips to initialize...'; sleep ${stagger}; "
        fi
        echo "source ~/tt-forge-fe/env/activate 2>/dev/null; \
            ${stagger_msg}\
            TT_VISIBLE_DEVICES=${chip_id} \
            TT_METAL_ARCH_NAME=blackhole \
            TT_MESH_GRAPH_DESC_PATH=${native_mesh} \
            python3 ${PROJECT_DIR}/compiletron.py run \
                --chip ${chip_id} --count ${COUNT}; \
            echo ''; echo '${done_msg}'; read -p 'Press Enter to close...'"
    fi
}

# ── Build deterministic tmux layout ──────────────────────────────────────────
#
# Uses pane IDs (%N) captured with -P at creation time.
# Pane IDs are globally unique — unaffected by base-index / pane-base-index.
#
# Split order (status strip first so it spans full width):
#   1. new-session           → P_TL  (full window)
#   2. split-window -v 15%   → P_STA (bottom strip, full width)
#   3. split-window -h 50%   → P_TR  (top-right, split from P_TL)
#   4. split-window -v 50%   → P_BL  (bottom-left, split from P_TL)
#   5. split-window -v 50%   → P_BR  (bottom-right, split from P_TR)

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

P_TL=$(tmux display-message -t "$SESSION" -p "#{pane_id}")
P_STA=$(tmux split-window -v -l 15% -t "$P_TL" -P -F "#{pane_id}")
P_TR=$(tmux split-window -h -l 50% -t "$P_TL" -P -F "#{pane_id}")
P_BL=$(tmux split-window -v -l 50% -t "$P_TL" -P -F "#{pane_id}")
P_BR=$(tmux split-window -v -l 50% -t "$P_TR" -P -F "#{pane_id}")

# ── Pane titles ───────────────────────────────────────────────────────────────

tmux select-pane -t "$P_TL"  -T "  Chip 0  "
tmux select-pane -t "$P_TR"  -T "  Chip 2  "
tmux select-pane -t "$P_BL"  -T "  Chip 1  "
tmux select-pane -t "$P_BR"  -T "  Chip 3  "
tmux select-pane -t "$P_STA" -T "  Status [$MODE]  "

tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "
tmux set -t "$SESSION" pane-border-style "fg=colour240"
tmux set -t "$SESSION" pane-active-border-style "fg=colour214,bold"

# ── Launch per-chip commands ──────────────────────────────────────────────────

tmux send-keys -t "$P_TL" "$(chip_cmd 0)" C-m
tmux send-keys -t "$P_TR" "$(chip_cmd 2)" C-m
tmux send-keys -t "$P_BL" "$(chip_cmd 1)" C-m
tmux send-keys -t "$P_BR" "$(chip_cmd 3)" C-m

# ── Status pane ───────────────────────────────────────────────────────────────

if [[ "$MODE" == "docker" ]]; then
    tmux send-keys -t "$P_STA" \
        "watch -n2 'echo \"Running containers:\"; docker ps --filter name=forge_chip --format \"  {{.Names}}  {{.Status}}\" 2>/dev/null || echo \"  (none)\"; echo \"\"; echo \"Done: \$(docker ps -a --filter name=forge_chip --filter status=exited --format x 2>/dev/null | wc -l)/4 chips\"'" \
        C-m
else
    tmux send-keys -t "$P_STA" \
        "watch -n2 'echo \"Mode: native | Run: ${TEST_NAME}\"; echo \"\"; ps aux | grep \"[c]ompiletron\" | awk \"{printf \\\"  Chip %s  %ss elapsed\\\\n\\\", \\\$NF, \\\$10}\" 2>/dev/null; echo \"\"; echo \"Forge processes: \$(pgrep -c python3 2>/dev/null || echo 0)\"'" \
        C-m
fi

# Focus top-left and attach
tmux select-pane -t "$P_TL"

echo ""
echo "  Mode: $MODE | Test: $TEST_NAME"
echo ""
echo "  ┌──────────────┬──────────────┐"
echo "  │  Chip 0      │  Chip 2      │"
echo "  ├──────────────┼──────────────┤"
echo "  │  Chip 1      │  Chip 3      │"
echo "  ├──────────────┴──────────────┤"
echo "  │  Status [$MODE]              │"
echo "  └─────────────────────────────┘"
echo ""
echo "  Ctrl+B + arrow keys = navigate panes"
echo "  Ctrl+B + D          = detach"
echo "  tmux attach -t $SESSION  = reattach"
echo ""

tmux attach-session -t "$SESSION"
