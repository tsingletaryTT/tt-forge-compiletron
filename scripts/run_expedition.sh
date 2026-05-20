#!/bin/bash
# scripts/run_expedition.sh
# Expedition Mode tmux layout — 4 chip panes + hardware monitor + status strip.
#
# Default layout (2×2):
#   ╔══════════════╦══════════════╗
#   ║  Chip 0      ║  Chip 1      ║
#   ╠══════════════╬══════════════╣
#   ║  Chip 2      ║  Chip 3      ║
#   ╠══════════════╩══════════════╣
#   ║  Status strip (5 rows)      ║
#   ╚═════════════════════════════╝
#
# --monitor layout (center column):
#   ╔═════════╦═══════════════╦═════════╗
#   ║  Chip 0 ║               ║  Chip 1 ║
#   ║         ║  tt-toplike   ║         ║
#   ╠═════════╣  --mode arcade╠═════════╣
#   ║  Chip 2 ║               ║  Chip 3 ║
#   ║         ║               ║         ║
#   ╠═════════╩═══════════════╩═════════╣
#   ║  Status strip (5 rows)            ║
#   ╚═══════════════════════════════════╝
#
# Key bindings added to the session:
#   prefix + B  →  Bestiary summary popup
#   prefix + G  →  Chip log viewer menu (piped logs)
#   prefix + M  →  Toggle monitor pane (tt-toplike) visibility
#
# Invoked by expedition.py with:
#   bash scripts/run_expedition.sh --chips N --run R [--monitor]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="expedition"
NUM_CHIPS=4
RUN_NUMBER=1
MONITOR=0
# Read ephemeral-cache flags from environment; default to off.
# expedition.py sets EXPEDITION_EPHEMERAL=1 and EXPEDITION_EVICT_FAILURES=1
# when the orchestrator decides workers should self-clean on exit.
EPHEMERAL=${EXPEDITION_EPHEMERAL:-0}
EVICT_FAILURES=${EXPEDITION_EVICT_FAILURES:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chips)   NUM_CHIPS="$2"; shift 2 ;;
        --run)     RUN_NUMBER="$2"; shift 2 ;;
        --monitor) MONITOR=1; shift ;;
        *)         shift ;;
    esac
done

if ! command -v tmux &>/dev/null; then
    echo "✗ tmux not installed: sudo apt install tmux"; exit 1
fi

if [[ ! -f ~/tt-forge-fe/env/activate ]]; then
    echo "✗ ~/tt-forge-fe/env/activate not found"; exit 1
fi

# Clear stale status and log files
rm -f /tmp/expedition_chip_{0,1,2,3}.status
rm -f /tmp/expedition_chip_{0,1,2,3}.log

# Purge stale forge shared memory segments from the previous run.
# forge.compile() creates sm_segment.tt-*.*.0 files in /dev/shm; a crashed
# worker leaves them behind and they corrupt subsequent forge.compile() calls.
find /dev/shm -name "sm_segment.tt-*.*.0" -delete 2>/dev/null || true

# ── Per-chip launcher scripts ─────────────────────────────────────────────────

# Build optional worker flags from env vars set by the orchestrator.
# These are evaluated here (outer shell) so the heredoc can interpolate them
# as plain strings — no backslash-escaping needed inside the generated script.
# Using $'...' quoting so the embedded newlines are real characters.
EXTRA_FLAGS=""
[ "${EPHEMERAL}" = "1" ]       && EXTRA_FLAGS="${EXTRA_FLAGS}"$' \\\n    --ephemeral'
[ "${EVICT_FAILURES}" = "1" ]  && EXTRA_FLAGS="${EXTRA_FLAGS}"$' \\\n    --evict-failures'

for chip_id in $(seq 0 $((NUM_CHIPS - 1))); do
    stagger=$((chip_id * 4))
    cat > "/tmp/expedition_chip_${chip_id}.sh" << CHIPSCRIPT
#!/bin/bash
clear
echo "┌─────────────────────────────────────"
echo "│  EXPEDITION  Chip ${chip_id}  Run #$(printf '%03d' ${RUN_NUMBER})"
echo "│"
echo ""

source ~/tt-forge-fe/env/activate
if [[ -z "\${TTFORGE_TOOLCHAIN_DIR}" && -z "\${TTMLIR_TOOLCHAIN_DIR}" ]]; then
    echo "ERROR: Forge env activation failed."
    read -rp "Press Enter to close..."
    exit 1
fi

STAGGER=${stagger}
if [[ \$STAGGER -gt 0 ]]; then
    echo "⏳ Staggered start: waiting \${STAGGER}s..."
    sleep \$STAGGER
fi

export TT_VISIBLE_DEVICES=${chip_id}
export TT_METAL_ARCH_NAME=blackhole
export TT_MESH_GRAPH_DESC_PATH=${PROJECT_DIR}/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto

python3 ${PROJECT_DIR}/lib/expedition/expedition_worker.py \
    --chip ${chip_id} \
    --run ${RUN_NUMBER} \
    --bestiary ${PROJECT_DIR}/data/bestiary.json \
    --queue /tmp/expedition_queue_chip${chip_id}.json \
    --results /tmp/expedition_results_chip${chip_id}.csv${EXTRA_FLAGS}
CHIPSCRIPT
    chmod +x "/tmp/expedition_chip_${chip_id}.sh"
done

# ── Build session ─────────────────────────────────────────────────────────────

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

P_TL=$(tmux display-message -t "$SESSION" -p "#{pane_id}")

if [[ "$MONITOR" -eq 1 && "$NUM_CHIPS" -ge 4 ]]; then
    # 3-column layout: chips left | tt-smi center | chips right
    # Step 1: carve off right column (~33% of window)
    P_TR=$(tmux split-window -h -l 33% -t "$P_TL" -P -F "#{pane_id}")
    # Step 2: carve monitor out of the remaining left 67% (~49% of 67% ≈ 33%)
    P_MON=$(tmux split-window -h -l 49% -t "$P_TL" -P -F "#{pane_id}")
    # Now: P_TL≈34% left | P_MON≈33% center | P_TR≈33% right
    P_BL=$(tmux split-window -v -l 50% -t "$P_TL" -P -F "#{pane_id}")
    P_BR=$(tmux split-window -v -l 50% -t "$P_TR" -P -F "#{pane_id}")
else
    # Standard 2×2 layout
    if [[ "$NUM_CHIPS" -ge 2 ]]; then
        P_TR=$(tmux split-window -h -l 50% -t "$P_TL" -P -F "#{pane_id}")
    fi
    if [[ "$NUM_CHIPS" -ge 3 ]]; then
        P_BL=$(tmux split-window -v -l 50% -t "$P_TL" -P -F "#{pane_id}")
    fi
    if [[ "$NUM_CHIPS" -ge 4 ]]; then
        P_BR=$(tmux split-window -v -l 50% -t "$P_TR" -P -F "#{pane_id}")
    fi
fi

# Full-width status strip — -f makes it span the entire window width so tmux
# rebalances height equally across all chip panes above it.
P_STA=$(tmux split-window -v -l 5 -f -t "$P_TL" -P -F "#{pane_id}")

# ── Session-wide appearance ───────────────────────────────────────────────────

# Kill tmux's own status bar — we have a dedicated strip pane for that.
tmux set -t "$SESSION" status off

# Mouse: click to focus panes, scroll history, resize pane borders by dragging.
tmux set -t "$SESSION" mouse on

# Double-line box drawing for borders (requires a font with Unicode box chars).
tmux set -t "$SESSION" pane-border-lines double

# Titles shown in the top border of each pane.
tmux set -t "$SESSION" pane-border-status top

# Active pane: bright teal (matches Tenstorrent palette #4FD1C5 → colour49).
# Inactive panes: muted blue-grey.
tmux set -t "$SESSION" pane-active-border-style "fg=colour49,bold"
tmux set -t "$SESSION" pane-border-style        "fg=colour238"

# Border label: teal + bold when active, dim when not.
tmux set -t "$SESSION" pane-border-format \
    "#{?pane_active,#[fg=colour49 bold],#[fg=colour240]} #{pane_title} #[default]"

# Chip panes survive process exit so you can read the final summary or crash
# output — prefix+x to manually close a pane when you're done.
for pane in "$P_TL" "$P_TR" "$P_BL" "$P_BR"; do
    [[ -n "$pane" ]] && tmux set-option -t "$pane" remain-on-exit on
done

# ── Pipe-pane logging — silent transcript of every chip pane ─────────────────
# Logs persist at /tmp/expedition_chip{N}.log for the duration of the session.

[[ -n "$P_TL" ]] && tmux pipe-pane -o -t "$P_TL" "cat >> /tmp/expedition_chip0.log"
[[ -n "$P_TR" ]] && tmux pipe-pane -o -t "$P_TR" "cat >> /tmp/expedition_chip1.log"
[[ -n "$P_BL" ]] && tmux pipe-pane -o -t "$P_BL" "cat >> /tmp/expedition_chip2.log"
[[ -n "$P_BR" ]] && tmux pipe-pane -o -t "$P_BR" "cat >> /tmp/expedition_chip3.log"

# ── Pane labels ───────────────────────────────────────────────────────────────

RUN_LABEL="Run #$(printf '%03d' "$RUN_NUMBER")"
tmux select-pane -t "$P_TL"  -T "  C0 · $RUN_LABEL  "
[[ -n "$P_TR"  ]] && tmux select-pane -t "$P_TR"  -T "  C1  "
[[ -n "$P_BL"  ]] && tmux select-pane -t "$P_BL"  -T "  C2  "
[[ -n "$P_BR"  ]] && tmux select-pane -t "$P_BR"  -T "  C3  "
[[ -n "$P_MON" ]] && tmux select-pane -t "$P_MON" -T "  Hardware  "
tmux select-pane -t "$P_STA" -T "  Score Board  "

# ── Session-scoped key bindings ───────────────────────────────────────────────

# prefix+B — Bestiary summary in a floating popup.
tmux bind-key -T prefix B \
    display-popup -w 80 -h 30 -b double \
    -T " ★ Expedition Bestiary " \
    -E "python3 '${PROJECT_DIR}/expedition.py' summary; echo; read -rp 'Press Enter...'"

# prefix+G — Interactive menu to tail a chip's piped log in a popup.
tmux bind-key -T prefix G \
    display-menu -T " Chip Logs " -x C -y C \
    "Chip 0" 0 "display-popup -w 90% -h 85% -b double -E 'less +GF /tmp/expedition_chip0.log'" \
    "Chip 1" 1 "display-popup -w 90% -h 85% -b double -E 'less +GF /tmp/expedition_chip1.log'" \
    "Chip 2" 2 "display-popup -w 90% -h 85% -b double -E 'less +GF /tmp/expedition_chip2.log'" \
    "Chip 3" 3 "display-popup -w 90% -h 85% -b double -E 'less +GF /tmp/expedition_chip3.log'"

# ── Launch workers ────────────────────────────────────────────────────────────

tmux send-keys -t "$P_TL" "bash /tmp/expedition_chip_0.sh" C-m
[[ -n "$P_TR"  ]] && tmux send-keys -t "$P_TR"  "bash /tmp/expedition_chip_1.sh" C-m
[[ -n "$P_BL"  ]] && tmux send-keys -t "$P_BL"  "bash /tmp/expedition_chip_2.sh" C-m
[[ -n "$P_BR"  ]] && tmux send-keys -t "$P_BR"  "bash /tmp/expedition_chip_3.sh" C-m

# Hardware monitor in center column (or right column for non-4-chip runs).
if [[ -n "$P_MON" ]]; then
    if command -v tt-toplike &>/dev/null; then
        tmux send-keys -t "$P_MON" "tt-toplike --mode arcade" C-m
    else
        tmux send-keys -t "$P_MON" \
            "echo 'tt-toplike not found — install tenstorrent-software-utils'" \
            C-m
    fi
fi

# Status strip — sub-second refresh via watch, no title bar (-t flag).
tmux send-keys -t "$P_STA" \
    "watch -n0.5 -t '${SCRIPT_DIR}/status_display.sh --expedition'" \
    C-m

# ── Attach ────────────────────────────────────────────────────────────────────

tmux select-pane -t "$P_TL"

echo ""
echo "  Expedition Run $RUN_LABEL — $NUM_CHIPS chip(s)$([ "$MONITOR" -eq 1 ] && echo " + hardware monitor")"
echo ""
echo "  prefix+arrows  navigate panes    prefix+B  bestiary popup"
echo "  prefix+G       chip log viewer   prefix+D  detach"
echo "  mouse          click / scroll / drag borders to resize"
echo ""
tmux attach-session -t "$SESSION"
