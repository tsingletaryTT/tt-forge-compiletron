#!/usr/bin/env python3
"""
compress_cast.py  --  Post-process an asciinema v2 cast file.

  1. Parses the file robustly using JSONDecoder.raw_decode() so that
     events whose terminal-output string contains a raw (unescaped)
     newline don't corrupt the line-by-line parse.
  2. Clamps idle gaps between events to MAX_IDLE seconds.
  3. Floors gaps below MIN_GAP seconds so burst events (many writes < 1ms
     apart) spread into smooth scrolling animation instead of a single-frame
     flash.  Without this, Textual's batched terminal writes create jerky
     playback where the screen jumps suddenly then freezes.
  4. Writes a clean, line-per-event output file.

Usage:
    python3 scripts/compress_cast.py INPUT.cast OUTPUT.cast [--max-idle SECS] [--min-gap SECS]
"""
import argparse, json, sys
from pathlib import Path


def parse_cast_robust(path: Path):
    """
    Read an asciinema v2 cast file and return (header_dict, list_of_events).

    Uses JSONDecoder.raw_decode to scan for complete JSON values in the raw
    text stream, so embedded bare newlines inside strings don't break
    parsing the way line-by-line iteration would.
    """
    text = path.read_bytes().decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    pos = 0
    header = None
    events = []

    while pos < len(text):
        # Skip whitespace (including newlines between events)
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            # Skip one character and try again (shouldn't normally happen
            # in a well-formed file, but guards against stray bytes)
            pos += 1
            continue

        if header is None:
            header = obj
        else:
            events.append(obj)
        pos = end

    return header, events


def compress(events, max_idle: float, min_gap: float = 0.0):
    """Clamp idle gaps > max_idle and floor gaps < min_gap.

    max_idle removes long dead zones (compilation pauses).
    min_gap spreads burst events — many writes < 1ms apart that the
    asciinema player would render as a single flash become a short
    animation instead, eliminating the jerky stop-motion look.
    """
    out = []
    out_t = 0.0  # running output timestamp (accounts for both adjustments)
    prev_in_t = 0.0  # previous input timestamp
    for ev in events:
        real_gap = ev[0] - prev_in_t
        # Clamp the gap to [min_gap, max_idle].
        adj_gap = min(max_idle, max(min_gap, real_gap)) if max_idle > 0 else max(min_gap, real_gap)
        out_t += adj_gap
        out.append([round(out_t, 4), ev[1], ev[2]])
        prev_in_t = ev[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input",  type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--max-idle", type=float, default=1.5,
                    help="Max gap between events in seconds (default 1.5)")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="Min gap between events in seconds (default 0 = off). "
                         "Use 0.02 to smooth out burst writes.")
    args = ap.parse_args()

    print(f"Parsing {args.input} …", file=sys.stderr)
    header, events = parse_cast_robust(args.input)
    if not events:
        print("ERROR: no events parsed", file=sys.stderr)
        sys.exit(1)

    original_dur = events[-1][0]
    print(f"  {len(events)} events, {original_dur:.1f}s raw", file=sys.stderr)

    compressed = compress(events, args.max_idle, args.min_gap)
    final_dur = compressed[-1][0]
    savings = original_dur - final_dur
    print(f"  → {len(compressed)} events, {final_dur:.1f}s compressed "
          f"({'saved' if savings >= 0 else 'added'} {abs(savings):.1f}s)", file=sys.stderr)
    print(f"  At 2x: {final_dur/2/60:.1f} min   "
          f"At 3x: {final_dur/3/60:.1f} min", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(json.dumps(header) + "\n")
        for ev in compressed:
            f.write(json.dumps(ev) + "\n")

    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
