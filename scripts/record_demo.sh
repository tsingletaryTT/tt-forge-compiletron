#!/usr/bin/env bash
# scripts/record_demo.sh
#
# Record a live expedition demo with asciinema.
# Uses tmux to automate the TUI keypress (Enter to start) so the
# recording is hands-free after you run this script.
#
# Requirements:
#   - asciinema installed  (sudo apt install asciinema)
#   - tmux installed       (sudo apt install tmux)
#   - forge env activated  (source ~/tt-forge-fe/env/activate)
#   - XLA venv exists      (~/tt-xla/venv)
#   - 4 Tenstorrent chips connected
#
# Output: docs/demo.cast   (replaces the generated placeholder)
#
# Usage:
#   cd /path/to/tt-forge-compiletron
#   source ~/tt-forge-fe/env/activate
#   bash scripts/record_demo.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CAST="docs/demo.cast"
SESSION="compiletron-demo-$$"
COLS=180
ROWS=48

# ── preflight ────────────────────────────────────────────────────────────────
for dep in asciinema tmux python3; do
    command -v "$dep" >/dev/null || { echo "ERROR: $dep not found"; exit 1; }
done

if ! python3 -c "import forge" &>/dev/null; then
    echo "WARNING: forge not importable — activate the forge environment first:"
    echo "  source ~/tt-forge-fe/env/activate"
fi

mkdir -p docs

# Kill any leftover session from a previous aborted run.
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "╔══════════════════════════════════════════════════════════════"
echo "║  TT-Forge Compiletron — Demo Recording"
echo "║"
echo "║  Terminal: ${COLS}×${ROWS}"
echo "║  Output:   $CAST"
echo "║  Run:      --seed-only --limit 4 --chips 4 --no-predownload"
echo "╚══════════════════════════════════════════════════════════════"
echo ""
echo "The TUI will launch automatically."
echo "  • Setup screen appears   → waits 8 seconds → presses Enter"
echo "  • Expedition runs        → 1 model per chip"
echo "  • Summary screen appears → recording stops after 15 seconds"
echo ""
echo "Press Ctrl-C to abort."
echo ""
sleep 2

# ── record ───────────────────────────────────────────────────────────────────
asciinema rec "$CAST" \
    --title "TT-Forge Compiletron — Expedition Demo" \
    --cols "$COLS" --rows "$ROWS" \
    --command "bash -c '
        # Create a detached tmux session at the right size.
        tmux new-session -d -s '"$SESSION"' -x $((COLS - 2)) -y $((ROWS - 2))

        # Launch the TUI in the session.
        tmux send-keys -t '"$SESSION"' \
            \"python3 expedition.py run --tui --seed-only --limit 4 --chips 4 --no-predownload\" \
            Enter

        # After 8 seconds the Setup screen will have rendered.
        # Send Enter to start the expedition, then q after completion.
        (
            sleep 8
            tmux send-keys -t '"$SESSION"' \"\" Enter
            # Wait long enough for 4 models to compile (max ~90s for Allam).
            sleep 120
            # Press q to quit from the Summary screen.
            tmux send-keys -t '"$SESSION"' \"q\" \"\"
        ) &
        HELPER_PID=\$!

        # Attach so asciinema captures the TUI output.
        tmux attach-session -t '"$SESSION"' || true

        kill \$HELPER_PID 2>/dev/null || true
    '"

echo ""
echo "Recording complete: $CAST"
echo ""
echo "Play back:   asciinema play $CAST"
echo "Upload:      asciinema upload $CAST"
echo ""
echo "To regenerate the scripted placeholder instead:"
echo "  python3 scripts/gen_demo_cast.py > docs/demo.cast"
