#!/bin/bash
# scripts/run_expedition.sh
# Expedition Mode tmux layout — 4 chip panes + shared status strip
#
# Layout (identical to run_4way_tmux.sh):
#   ┌──────────────┬──────────────┐
#   │  Chip 0      │  Chip 1      │
#   ├──────────────┼──────────────┤
#   │  Chip 2      │  Chip 3      │
#   ├──────────────┴──────────────┤
#   │  Status (scores, streaks)   │
#   └─────────────────────────────┘
#
# Invoked by expedition.py with:
#   bash scripts/run_expedition.sh --chips N --run R

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="expedition"
NUM_CHIPS=4
RUN_NUMBER=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chips) NUM_CHIPS="$2"; shift 2 ;;
        --run)   RUN_NUMBER="$2"; shift 2 ;;
        *)       shift ;;
    esac
done

if ! command -v tmux &>/dev/null; then
    echo "✗ tmux not installed: sudo apt install tmux"; exit 1
fi

if [[ ! -f ~/tt-forge-fe/env/activate ]]; then
    echo "✗ ~/tt-forge-fe/env/activate not found"; exit 1
fi

# Clear stale expedition status files
rm -f /tmp/expedition_chip_{0,1,2,3}.status

# Write per-chip launcher scripts to /tmp
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
    --results /tmp/expedition_results_chip${chip_id}.csv
CHIPSCRIPT
    chmod +x "/tmp/expedition_chip_${chip_id}.sh"
done

# Build tmux layout
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

P_TL=$(tmux display-message -t "$SESSION" -p "#{pane_id}")
P_STA=$(tmux split-window -v -l 6 -t "$P_TL" -P -F "#{pane_id}")

if [[ "$NUM_CHIPS" -ge 2 ]]; then
    P_TR=$(tmux split-window -h -l 50% -t "$P_TL" -P -F "#{pane_id}")
fi
if [[ "$NUM_CHIPS" -ge 3 ]]; then
    P_BL=$(tmux split-window -v -l 50% -t "$P_TL" -P -F "#{pane_id}")
fi
if [[ "$NUM_CHIPS" -ge 4 ]]; then
    P_BR=$(tmux split-window -v -l 50% -t "$P_TR" -P -F "#{pane_id}")
fi

# Pane titles
tmux select-pane -t "$P_TL"  -T "  Chip 0 — Expedition #$(printf '%03d' $RUN_NUMBER)  "
[[ -n "$P_TR"  ]] && tmux select-pane -t "$P_TR"  -T "  Chip 1  "
[[ -n "$P_BL"  ]] && tmux select-pane -t "$P_BL"  -T "  Chip 2  "
[[ -n "$P_BR"  ]] && tmux select-pane -t "$P_BR"  -T "  Chip 3  "
tmux select-pane -t "$P_STA" -T "  Score Board  "

tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "
tmux set -t "$SESSION" pane-border-style "fg=colour240"
tmux set -t "$SESSION" pane-active-border-style "fg=colour214,bold"

# Launch workers
tmux send-keys -t "$P_TL" "bash /tmp/expedition_chip_0.sh" C-m
[[ -n "$P_TR"  ]] && tmux send-keys -t "$P_TR"  "bash /tmp/expedition_chip_1.sh" C-m
[[ -n "$P_BL"  ]] && tmux send-keys -t "$P_BL"  "bash /tmp/expedition_chip_2.sh" C-m
[[ -n "$P_BR"  ]] && tmux send-keys -t "$P_BR"  "bash /tmp/expedition_chip_3.sh" C-m

# Status strip — reads expedition_chip_N.status files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmux send-keys -t "$P_STA" \
    "watch -n1 -t '${SCRIPT_DIR}/status_display.sh --expedition'" \
    C-m

tmux select-pane -t "$P_TL"
echo ""
echo "  Expedition Run #$(printf '%03d' $RUN_NUMBER) — $NUM_CHIPS chip(s)"
echo ""
echo "  Ctrl+B + arrow = navigate  |  Ctrl+B + D = detach"
echo ""
tmux attach-session -t "$SESSION"
