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
    # Prefer native if the Forge environment is available; fall back to Docker.
    if [[ -f ~/tt-forge-fe/env/activate ]]; then
        MODE="native"
    elif docker image inspect "$DOCKER_IMAGE" &>/dev/null 2>&1; then
        MODE="docker"
    else
        MODE="native"  # Let pre-flight checks report the real error
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

# ── Write per-chip launcher scripts (native mode) ────────────────────────────
#
# Avoids quoting hell in tmux send-keys by writing real bash scripts to /tmp/.
# Each pane runs: bash /tmp/forge_chip_N.sh
# This also makes errors fully visible (no 2>/dev/null suppression).

write_native_scripts() {
    local native_mesh="${PROJECT_DIR}/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
    for chip_id in 0 1 2 3; do
        local stagger=$((chip_id * 4))
        cat > "/tmp/forge_chip_${chip_id}.sh" << CHIPSCRIPT
#!/bin/bash
clear
echo "┌─────────────────────────────────────"
echo "│  TT-Forge Chip ${chip_id}  (native mode)"
echo "│"
echo ""

# Activate forge environment
source ~/tt-forge-fe/env/activate
if [[ -z "\${TTFORGE_TOOLCHAIN_DIR}" && -z "\${TTMLIR_TOOLCHAIN_DIR}" ]]; then
    echo "ERROR: Forge env activation failed."
    echo "  Tried:  source ~/tt-forge-fe/env/activate"
    echo "  Wanted: TTFORGE_TOOLCHAIN_DIR or TTMLIR_TOOLCHAIN_DIR to be set"
    read -rp "Press Enter to close..."
    exit 1
fi
echo "✓ Forge env: \${TTFORGE_TOOLCHAIN_DIR:-\${TTMLIR_TOOLCHAIN_DIR}}"

# Staggered startup: chip_id * 4s so UMD initializes sequentially
# (all 4 chips racing to open PCIe devices simultaneously causes lock failures)
STAGGER=${stagger}
if [[ \$STAGGER -gt 0 ]]; then
    echo ""
    echo "⏳ Staggered start: waiting \${STAGGER}s for earlier chips to initialize..."
    sleep \$STAGGER
fi

echo ""
export TT_VISIBLE_DEVICES=${chip_id}
export TT_METAL_ARCH_NAME=blackhole
export TT_MESH_GRAPH_DESC_PATH=${native_mesh}
echo "  Chip:       ${chip_id}"
echo "  Stride:     4 (round-robin across all 4 chips)"
echo "  Mesh:       p100"
echo ""

# lib/worker.py runs the full visual pipeline:
#   - pyfiglet ASCII art banners with rotating Tenstorrent colors
#   - [1/3][2/3][3/3] steps, 3s celebration pause per model
#   - Round-robin: chip ${chip_id} compiles models ${chip_id}, $((chip_id+4)), $((chip_id+8)), ...
#   - Colored ✓/✗ checklist on completion, then stays alive
python3 ${PROJECT_DIR}/lib/worker.py --chip ${chip_id} --stride 4 \
    --results /tmp/forge_results_chip_${chip_id}.csv
CHIPSCRIPT
        chmod +x "/tmp/forge_chip_${chip_id}.sh"
    done
}

# ── Build per-chip tmux command ───────────────────────────────────────────────

chip_cmd() {
    local chip_id=$1
    if [[ "$MODE" == "docker" ]]; then
        # --entrypoint python3 bypasses docker-entrypoint.sh, which otherwise
        # receives 'python3' as its $1 command, hits the '*' fallback, and runs
        # "python3 compiletron.py python3 ..." — passing python3 as a subcommand.
        echo "docker run --rm \
            --name forge_chip_${chip_id} \
            --device=/dev/tenstorrent:/dev/tenstorrent \
            --shm-size=16g \
            --entrypoint python3 \
            -e TT_VISIBLE_DEVICES=${chip_id} \
            -e TT_METAL_ARCH_NAME=blackhole \
            -e TT_MESH_GRAPH_DESC_PATH=${MESH_DESC_PATH} \
            ${DOCKER_IMAGE} \
            /app/scripts/docker/forge_worker.py ${TEST_NAME}; \
            echo ''; echo '════ CHIP ${chip_id} DONE ════'; read -p 'Press Enter to close...'"
    else
        echo "bash /tmp/forge_chip_${chip_id}.sh"
    fi
}

# ── Build deterministic tmux layout ──────────────────────────────────────────
#
# Uses pane IDs (%N) captured with -P at creation time.
# Pane IDs are globally unique — unaffected by base-index / pane-base-index.
#
# Split order (status strip first so it spans full width):
#   1. new-session           → P_TL  (full window)
#   2. split-window -v 5     → P_STA (bottom strip, 5 lines, full width)
#   3. split-window -h 50%   → P_TR  (top-right, split from P_TL)
#   4. split-window -v 50%   → P_BL  (bottom-left, split from P_TL)
#   5. split-window -v 50%   → P_BR  (bottom-right, split from P_TR)

[[ "$MODE" == "native" ]] && write_native_scripts

# Clear status files from any previous run so the status pane starts zeroed
rm -f /tmp/compiletron_chip_{0,1,2,3}.status

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

P_TL=$(tmux display-message -t "$SESSION" -p "#{pane_id}")
P_STA=$(tmux split-window -v -l 5 -t "$P_TL" -P -F "#{pane_id}")
P_TR=$(tmux split-window -h -l 50% -t "$P_TL" -P -F "#{pane_id}")
P_BL=$(tmux split-window -v -l 50% -t "$P_TL" -P -F "#{pane_id}")
P_BR=$(tmux split-window -v -l 50% -t "$P_TR" -P -F "#{pane_id}")

# ── Pane titles ───────────────────────────────────────────────────────────────

tmux select-pane -t "$P_TL"  -T "  Chip 0  "
tmux select-pane -t "$P_TR"  -T "  Chip 1  "
tmux select-pane -t "$P_BL"  -T "  Chip 2  "
tmux select-pane -t "$P_BR"  -T "  Chip 3  "
tmux select-pane -t "$P_STA" -T "  Status [$MODE]  "

tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "
tmux set -t "$SESSION" pane-border-style "fg=colour240"
tmux set -t "$SESSION" pane-active-border-style "fg=colour214,bold"

# ── Launch per-chip commands ──────────────────────────────────────────────────

tmux send-keys -t "$P_TL" "$(chip_cmd 0)" C-m
tmux send-keys -t "$P_TR" "$(chip_cmd 1)" C-m
tmux send-keys -t "$P_BL" "$(chip_cmd 2)" C-m
tmux send-keys -t "$P_BR" "$(chip_cmd 3)" C-m

# ── Status pane ───────────────────────────────────────────────────────────────
# Renders live ASCII progress bars for all 4 chips using status files written
# by lib/worker.py to /tmp/compiletron_chip_N.status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmux send-keys -t "$P_STA" \
    "watch -n1 -t '${SCRIPT_DIR}/status_display.sh'" \
    C-m

# Focus top-left and attach
tmux select-pane -t "$P_TL"

echo ""
echo "  Mode: $MODE | Test: $TEST_NAME"
echo ""
echo "  ┌──────────────┬──────────────"
echo "  │  Chip 0      │  Chip 1"
echo "  ├──────────────┼──────────────"
echo "  │  Chip 2      │  Chip 3"
echo "  ├──────────────┴──────────────"
echo "  │  Status [$MODE]"
echo ""
echo "  Ctrl+B + arrow keys = navigate panes"
echo "  Ctrl+B + D          = detach"
echo "  tmux attach -t $SESSION  = reattach"
echo ""

tmux attach-session -t "$SESSION"
