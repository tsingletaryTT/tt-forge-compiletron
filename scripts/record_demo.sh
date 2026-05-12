#!/usr/bin/env bash
# scripts/record_demo.sh
#
# Record a live expedition demo with asciinema.
#
# The TUI SetupScreen auto-starts after 4 seconds if Enter isn't pressed,
# so no tmux key injection is needed — just launch asciinema directly.
# The recording ends when the user presses q on the SummaryScreen, or
# when asciinema's command exits naturally.
#
# Requirements:
#   - asciinema installed  (sudo apt install asciinema)
#   - forge env activated  (source ~/tt-forge-fe/env/activate)
#   - 4 Tenstorrent chips connected
#
# Output: docs/demo_raw.cast (compress separately with compress_cast.py)
#
# Usage:
#   cd /path/to/tt-forge-compiletron
#   source ~/tt-forge-fe/env/activate
#   bash scripts/record_demo.sh [--models N]   # default 4 per chip = 16 total

set -euo pipefail
cd "$(dirname "$0")/.."

CAST="docs/demo_raw.cast"
COLS=220
ROWS=58
CHIPS=4
MODELS_PER_CHIP=${MODELS_PER_CHIP:-4}

# Allow --models N override
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS_PER_CHIP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TOTAL=$(( CHIPS * MODELS_PER_CHIP ))

# ── preflight ────────────────────────────────────────────────────────────────
for dep in asciinema python3; do
    command -v "$dep" >/dev/null || { echo "ERROR: $dep not found"; exit 1; }
done

if ! python3 -c "import forge" &>/dev/null; then
    echo "WARNING: forge not importable — activate the forge environment first:"
    echo "  source ~/tt-forge-fe/env/activate"
fi

mkdir -p docs

echo "╔══════════════════════════════════════════════════════════════"
echo "║  TT-Forge Compiletron — Demo Recording"
echo "║"
echo "║  Terminal: ${COLS}×${ROWS}"
echo "║  Output:   $CAST"
echo "║  Run:      --seed-only --limit ${TOTAL} --chips ${CHIPS} --no-predownload"
echo "║            (${MODELS_PER_CHIP} models per chip)"
echo "╚══════════════════════════════════════════════════════════════"
echo ""
echo "The TUI auto-starts after 4 seconds (or press Enter immediately)."
echo "Press q on the Summary screen to end the recording."
echo ""
echo "Press Ctrl-C to abort."
echo ""
sleep 2

# ── record ───────────────────────────────────────────────────────────────────
asciinema rec "$CAST" \
    --overwrite \
    --title "TT-Forge Compiletron — Expedition Demo" \
    --cols "$COLS" --rows "$ROWS" \
    --command "python3 expedition.py run --tui \
        --seed-only \
        --limit ${TOTAL} \
        --chips ${CHIPS} \
        --no-predownload"

echo ""
echo "Raw recording: $CAST"
echo ""
echo "Post-process (smooth + compress):"
echo "  python3 scripts/compress_cast.py $CAST docs/demo.cast --max-idle 1.2 --min-gap 0.02"
echo ""
echo "Play back raw:  asciinema play $CAST"
