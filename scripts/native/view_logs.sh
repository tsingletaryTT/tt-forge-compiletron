#!/bin/bash
# View parallel Forge compilation logs in tmux
# Auto-scales layout based on number of chips (1-32+)

SESSION="forge-compiletron"
LOG_DIR="/tmp/forge_compiletron_logs"

# Detect number of chips
NUM_CHIPS=$(python3 -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname('$0'), '..', 'lib'))
from hardware import detect_hardware
hw = detect_hardware()
print(hw.get('num_chips', 0))
" 2>/dev/null || echo "4")

if [ "$NUM_CHIPS" -eq 0 ]; then
    echo "❌ No chips detected, defaulting to 4-chip layout"
    NUM_CHIPS=4
fi

echo "Creating tmux session for $NUM_CHIPS chip(s)..."

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null

# Create log directory
mkdir -p "$LOG_DIR"

# Create empty log files
for ((i=0; i<NUM_CHIPS; i++)); do
    touch "$LOG_DIR/chip${i}.log"
done

# Calculate grid layout (roughly square)
COLS=$(python3 -c "import math; print(math.ceil(math.sqrt($NUM_CHIPS)))")
ROWS=$(python3 -c "import math; print(math.ceil($NUM_CHIPS / $COLS))")

echo "Layout: ${ROWS}x${COLS} grid"

# Create tmux session
tmux new-session -d -s "$SESSION"

# Create panes based on layout
PANE=0
for ((row=0; row<ROWS; row++)); do
    for ((col=0; col<COLS; col++)); do
        CHIP_ID=$((row * COLS + col))

        if [ "$CHIP_ID" -ge "$NUM_CHIPS" ]; then
            break 2
        fi

        if [ "$PANE" -eq 0 ]; then
            # First pane already exists
            PANE=1
        else
            # Split to create new pane
            if [ "$col" -eq 0 ]; then
                # New row - split vertically from pane 0
                tmux split-window -v -t "$SESSION.0"
            else
                # Same row - split horizontally
                tmux split-window -h
            fi
        fi

        # Start log viewer in this pane
        tmux send-keys -t "$SESSION.$CHIP_ID" \
            "echo '═══ CHIP $CHIP_ID LOG ═══' && tail -f $LOG_DIR/chip${CHIP_ID}.log" C-m
    done
done

# Balance layout
tmux select-layout -t "$SESSION" tiled

echo ""
echo "✓ Tmux session '$SESSION' ready!"
echo ""
echo "Layout: ${ROWS}x${COLS} for $NUM_CHIPS chip(s)"
echo "Logs: $LOG_DIR/chip*.log"
echo ""
echo "Controls:"
echo "  • Switch panes: Ctrl+b then arrow keys"
echo "  • Detach: Ctrl+b then d"
echo "  • Reattach: tmux attach -t $SESSION"
echo "  • Kill: tmux kill-session -t $SESSION"
echo ""
echo "Attaching in 2 seconds..."
sleep 2

tmux attach-session -t "$SESSION"
