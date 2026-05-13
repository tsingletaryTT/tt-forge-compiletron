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
#   bash scripts/record_demo.sh --curated      # curated showcase queue (website demo)

set -euo pipefail
cd "$(dirname "$0")/.."

CAST="docs/demo_raw.cast"
COLS=220
ROWS=50
CHIPS=4
MODELS_PER_CHIP=${MODELS_PER_CHIP:-4}
AUTO_QUIT=30   # seconds to linger on summary screen before auto-exit
CURATED=0      # use curated demo queue (hand-picked showcase models)
TUI=1          # 1=TUI mode, 0=scrolling terminal mode

# Allow --models N, --auto-quit N, --curated, --no-tui overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)    MODELS_PER_CHIP="$2"; shift 2 ;;
        --auto-quit) AUTO_QUIT="$2";       shift 2 ;;
        --no-auto)   AUTO_QUIT=0;          shift   ;;
        --curated)   CURATED=1;            shift   ;;
        --no-tui)    TUI=0;                shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Use separate output file for scrolling mode
[[ "$TUI" -eq 0 ]] && CAST="docs/demo_scroll_raw.cast"

TOTAL=$(( CHIPS * MODELS_PER_CHIP ))

# ── preflight ────────────────────────────────────────────────────────────────
for dep in asciinema python3; do
    command -v "$dep" >/dev/null || { echo "ERROR: $dep not found"; exit 1; }
done

if ! python3 -c "import forge" &>/dev/null; then
    echo "WARNING: forge not importable — activate the forge environment first:"
    echo "  source ~/tt-forge-fe/env/activate"
fi

# ── shm cleanup ──────────────────────────────────────────────────────────────
# Forge compile leaves behind sm_segment.* and tt_device_*_memory files in
# /dev/shm after each run.  Stale segments from a previous (possibly crashed)
# run will cause the next forge.compile() call to hang indefinitely.
SHM_COUNT=$(find /dev/shm -maxdepth 1 \( -name 'sm_segment.tt-quietbox.*.0' -o -name 'tt_device_*_memory' \) 2>/dev/null | wc -l)
if [[ "$SHM_COUNT" -gt 0 ]]; then
    echo "⚠  Clearing $SHM_COUNT stale /dev/shm file(s) from previous run..."
    find /dev/shm -maxdepth 1 -name 'sm_segment.tt-quietbox.*.0' -delete
    find /dev/shm -maxdepth 1 -name 'tt_device_*_memory' -delete
fi

mkdir -p docs

if [[ "$AUTO_QUIT" -gt 0 ]]; then
    AUTO_QUIT_LABEL="auto-exit ${AUTO_QUIT}s after summary"
else
    AUTO_QUIT_LABEL="manual — press q on summary screen to stop"
fi

if [[ "$CURATED" -eq 1 ]]; then
    RUN_DESC="--curated  (AlexNet → GPT-2 → BEiT → DenseUNet FAIL → BLOOM 4-chip finale)"
else
    RUN_DESC="--seed-only --limit ${TOTAL} --chips ${CHIPS} --no-predownload (${MODELS_PER_CHIP} models per chip)"
fi

MODE_LABEL=$([[ "$TUI" -eq 1 ]] && echo "TUI" || echo "Scrolling (no TUI)")
echo "╔══════════════════════════════════════════════════════════════"
echo "║  TT-Forge Compiletron — Demo Recording"
echo "║"
echo "║  Terminal: ${COLS}×${ROWS}   Mode: ${MODE_LABEL}"
echo "║  Output:   $CAST"
echo "║  Run:      ${RUN_DESC}"
echo "║  Finish:   ${AUTO_QUIT_LABEL}"
echo "╚══════════════════════════════════════════════════════════════"
echo ""
echo "The TUI auto-starts after 4 seconds (or press Enter immediately)."
if [[ "$AUTO_QUIT" -gt 0 ]]; then
    echo "Recording ends automatically ${AUTO_QUIT}s after the summary screen appears."
else
    echo "Press q on the Summary screen to end the recording."
fi
echo ""
echo "Press Ctrl-C to abort."
echo ""
sleep 2

# ── record ───────────────────────────────────────────────────────────────────
AUTO_QUIT_FLAG=""
[[ "$AUTO_QUIT" -gt 0 ]] && AUTO_QUIT_FLAG="--auto-quit ${AUTO_QUIT}"

TUI_FLAG=""
[[ "$TUI" -eq 1 ]] && TUI_FLAG="--tui"

if [[ "$CURATED" -eq 1 ]]; then
    EXPEDITION_CMD="python3 expedition.py run ${TUI_FLAG} --curated ${AUTO_QUIT_FLAG}"
else
    EXPEDITION_CMD="python3 expedition.py run ${TUI_FLAG} \
        --seed-only \
        --limit ${TOTAL} \
        --chips ${CHIPS} \
        --no-predownload \
        ${AUTO_QUIT_FLAG}"
fi

asciinema rec "$CAST" \
    --overwrite \
    --title "TT-Forge Compiletron — Expedition Demo" \
    --cols "$COLS" --rows "$ROWS" \
    --command "${EXPEDITION_CMD}"

echo ""
COMPRESSED="${CAST/_raw/}"
echo "Raw recording: $CAST"
echo ""
echo "Post-process (smooth + compress):"
echo "  python3 scripts/compress_cast.py $CAST ${COMPRESSED} --max-idle 1.2 --min-gap 0.02"
echo ""
echo "Play back raw:  asciinema play $CAST"
