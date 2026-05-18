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
BENCH_PASSES=0 # 0 = no bench; N = run N timed inference passes per model

usage() {
    cat <<EOF
Usage: bash scripts/record_demo.sh [OPTIONS]

Record a live tt-forge-compiletron expedition demo with asciinema.

Options:
  --curated              Use the hand-curated showcase queue:
                           AlexNet → GPT-2 → BEiT → DenseUNet FAIL → BLOOM 4-chip finale
                         (default: random seed models, MODELS_PER_CHIP per chip)
  --models N             Models per chip for the random queue (default: ${MODELS_PER_CHIP})
  --bench                Enable benchmarking: 5 timed inference passes per model
  --bench-passes N       Set exact number of bench passes (implies --bench)
  --no-tui               Scrolling terminal output instead of TUI
  --auto-quit N          Seconds to linger on summary before auto-exit (default: ${AUTO_QUIT})
  --no-auto              Disable auto-quit; press q manually on the summary screen
  -h, --help             Show this help and exit

Output files:
  TUI mode:              docs/demo_raw.cast
  --no-tui:              docs/demo_scroll_raw.cast
  --bench / --bench-passes: docs/demo_bench_raw.cast

Post-processing (run after recording):
  python3 scripts/compress_cast.py docs/demo_raw.cast docs/demo.cast --max-idle 1.2 --min-gap 0.02

Common invocations:
  bash scripts/record_demo.sh --curated --bench     # showcase + benchmarks (recommended)
  bash scripts/record_demo.sh --curated             # showcase, no bench
  bash scripts/record_demo.sh --models 6            # 6 random models per chip
  bash scripts/record_demo.sh --no-tui --curated    # scrolling output for debugging
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)      usage; exit 0 ;;
        --models)       MODELS_PER_CHIP="$2"; shift 2 ;;
        --auto-quit)    AUTO_QUIT="$2";       shift 2 ;;
        --no-auto)      AUTO_QUIT=0;          shift   ;;
        --curated)      CURATED=1;            shift   ;;
        --bench)        BENCH_PASSES=5;       shift   ;;
        --bench-passes) BENCH_PASSES="$2";    shift 2 ;;
        --no-tui)       TUI=0;                shift   ;;
        *) echo "Unknown arg: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

# Use separate output files for scrolling / bench modes
[[ "$TUI" -eq 0 ]] && CAST="docs/demo_scroll_raw.cast"
[[ "$BENCH_PASSES" -gt 0 ]] && CAST="docs/demo_bench_raw.cast"

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

BENCH_LABEL=""
[[ "$BENCH_PASSES" -gt 0 ]] && BENCH_LABEL="--bench-passes ${BENCH_PASSES}  (2 warm-up + ${BENCH_PASSES} timed passes per model)"

MODE_LABEL=$([[ "$TUI" -eq 1 ]] && echo "TUI" || echo "Scrolling (no TUI)")
echo "╔══════════════════════════════════════════════════════════════"
echo "║  TT-Forge Compiletron — Demo Recording"
echo "║"
echo "║  Terminal: ${COLS}×${ROWS}   Mode: ${MODE_LABEL}"
echo "║  Output:   $CAST"
echo "║  Run:      ${RUN_DESC}"
[[ -n "$BENCH_LABEL" ]] && echo "║  Bench:    ${BENCH_LABEL}"
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

BENCH_FLAG=""
[[ "$BENCH_PASSES" -gt 0 ]] && BENCH_FLAG="--bench-passes ${BENCH_PASSES}"

if [[ "$CURATED" -eq 1 ]]; then
    EXPEDITION_CMD="python3 expedition.py run ${TUI_FLAG} --curated ${AUTO_QUIT_FLAG} ${BENCH_FLAG}"
else
    EXPEDITION_CMD="python3 expedition.py run ${TUI_FLAG} \
        --seed-only \
        --limit ${TOTAL} \
        --chips ${CHIPS} \
        --no-predownload \
        ${AUTO_QUIT_FLAG} ${BENCH_FLAG}"
fi

# When bench passes are enabled, show stats inside the recording after the run.
if [[ "$BENCH_PASSES" -gt 0 ]]; then
    STATS_CMD="python3 scripts/show_perf_stats.py"
    FULL_CMD="${EXPEDITION_CMD} ; ${STATS_CMD}"
else
    FULL_CMD="${EXPEDITION_CMD}"
fi

asciinema rec "$CAST" \
    --overwrite \
    --title "TT-Forge Compiletron — Expedition Demo" \
    --cols "$COLS" --rows "$ROWS" \
    --command "bash -c '${FULL_CMD}'"

echo ""
COMPRESSED="${CAST/_raw/}"
echo "Raw recording: $CAST"
echo ""
echo "Post-process (smooth + compress):"
echo "  python3 scripts/compress_cast.py $CAST ${COMPRESSED} --max-idle 1.2 --min-gap 0.02"
echo ""
echo "Play back raw:  asciinema play $CAST"
