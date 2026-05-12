#!/usr/bin/env bash
# scripts/render_demo_video.sh
#
# Renders docs/demo.cast to a true 24-bit H.264 MP4 via:
#   Xvfb  →  xterm (Ubuntu Mono, dark theme)  →  ffmpeg x11grab
#
# No GIF step — full color terminal rendering.
#
# Output: docs/demo.mp4
#
# Usage:
#   cd /path/to/tt-forge-compiletron
#   bash scripts/render_demo_video.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CAST="docs/demo.cast"
OUT="docs/demo.mp4"
DISPLAY_NUM=":99"
SPEED=2
FONT="Ubuntu Mono"
FONT_SIZE=13   # pt — tuned so 220x58 terminal ≈ 1920×1050 px
BG="#0F2A35"
FG="#E8F0F2"
COLS=220
ROWS=58

# ── preflight ─────────────────────────────────────────────────────────────────
for dep in Xvfb xterm ffmpeg asciinema; do
    command -v "$dep" >/dev/null || { echo "ERROR: $dep not found"; exit 1; }
done
[[ -f "$CAST" ]] || { echo "ERROR: $CAST not found — run compress_cast.py first"; exit 1; }

CAST_DUR=$(python3 -c "
import json
events=[]
with open('$CAST') as f:
    f.readline()
    for l in f: events.append(json.loads(l.strip()))
print(events[-1][0])
")
RECORD_SECS=$(python3 -c "print(int($CAST_DUR / $SPEED + 6))")
echo "Cast duration: ${CAST_DUR}s  →  recording ${RECORD_SECS}s at ${SPEED}x"

# ── kill any leftover Xvfb on :99 ─────────────────────────────────────────────
pkill -f "Xvfb $DISPLAY_NUM" 2>/dev/null || true
sleep 0.3

# ── start virtual display ─────────────────────────────────────────────────────
Xvfb "$DISPLAY_NUM" -screen 0 4096x2160x24 &
XVFB_PID=$!
sleep 0.8

# ── launch xterm, get its pixel geometry ──────────────────────────────────────
DISPLAY="$DISPLAY_NUM" xterm \
    -geometry "${COLS}x${ROWS}+0+0" \
    -fa "$FONT" -fs "$FONT_SIZE" \
    -bg "$BG" -fg "$FG" \
    -bc -cr "#4FD1C5" \
    -title "compiletron-capture" \
    -T   "compiletron-capture" \
    -e bash -c "asciinema play --speed $SPEED '$CAST'; sleep 4" &
XTERM_PID=$!
sleep 1.5   # let xterm render its first frame

WIN_GEOM=$(DISPLAY="$DISPLAY_NUM" xwininfo -name "compiletron-capture" 2>/dev/null | \
    awk '/Width:/{w=$2} /Height:/{h=$2}
         /Absolute upper-left X:/{x=$NF}
         /Absolute upper-left Y:/{y=$NF}
         END{print w"x"h"+"x"+"y}')
WIN_WH=$(echo "$WIN_GEOM" | cut -d+ -f1)
WIN_X=$(echo "$WIN_GEOM"  | cut -d+ -f2)
WIN_Y=$(echo "$WIN_GEOM"  | cut -d+ -f3)

# Force even dimensions (H.264 requires width/height divisible by 2)
W=$(echo "$WIN_WH" | cut -dx -f1)
H=$(echo "$WIN_WH" | cut -dx -f2)
W=$(( (W / 2) * 2 ))
H=$(( (H / 2) * 2 ))

echo "xterm window: ${W}x${H} at offset ${WIN_X},${WIN_Y}"

# ── ffmpeg: capture window at 30fps ───────────────────────────────────────────
rm -f "$OUT"
DISPLAY="$DISPLAY_NUM" ffmpeg -y \
    -f x11grab \
    -framerate 30 \
    -video_size "${W}x${H}" \
    -i "${DISPLAY_NUM}+${WIN_X},${WIN_Y}" \
    -t "$RECORD_SECS" \
    -c:v libx264 \
    -preset slow \
    -crf 15 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUT" 2>&1

# ── cleanup ───────────────────────────────────────────────────────────────────
kill "$XTERM_PID" 2>/dev/null || true
kill "$XVFB_PID"  2>/dev/null || true

echo ""
ls -lh "$OUT"
echo "Done. Embed with:"
echo "  <video autoplay loop muted playsinline src=\"demo.mp4\"></video>"
