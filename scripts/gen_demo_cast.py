#!/usr/bin/env python3
"""
Generate a demo asciinema v2 cast for tt-forge-compiletron.

Simulates a 4-chip expedition run: forge (chips 0-2) and XLA/JAX (chip 3).
Shows rarity reveals, scoring, First Voice, and the summary screen.

Usage:
    python3 scripts/gen_demo_cast.py > docs/demo.cast
"""
import json
import sys

COLS = 180
ROWS = 48

# ── ANSI palette (matches expedition workers) ────────────────────────────────
T   = "\033[38;5;87m"    # teal
G   = "\033[38;5;220m"   # gold
R   = "\033[38;5;196m"   # red
GN  = "\033[38;5;46m"    # green
PK  = "\033[38;5;213m"   # pink
BL  = "\033[38;5;75m"    # blue
CY  = "\033[38;5;51m"    # bright cyan (XLA)
DIM = "\033[2m"
B   = "\033[1m"
RS  = "\033[0m"

events: list = []
_t: float = 0.0


def emit(text: str, after: float = 0.0) -> None:
    global _t
    # Split into per-character events for typewriter effect on prompts.
    events.append([round(_t, 3), "o", text])
    _t += after


def line(text: str = "", after: float = 0.04) -> None:
    emit(text + "\r\n", after)


def pause(secs: float) -> None:
    global _t
    _t += secs


def blank(n: int = 1, after: float = 0.02) -> None:
    for _ in range(n):
        line("", after)


def sep(char: str = "─", width: int = COLS - 2, color: str = DIM, after: float = 0.02) -> None:
    line(f"{color}{char * width}{RS}", after)


def thick_sep(color: str = T, after: float = 0.04) -> None:
    line(f"{B}{color}{'═' * (COLS - 2)}{RS}", after)


def chip_banner(chip_id: int, backend: str, run: int = 7) -> None:
    col = CY if backend == "xla" else T
    label = f"XLA" if backend == "xla" else "FORGE"
    blank(1, 0)
    thick_sep(col, 0.01)
    line(f"{B}{col}  ⚡ EXPEDITION {label} CHIP {chip_id}  ·  run #{run:03d}{RS}", 0.03)
    thick_sep(col, 0.04)
    blank(1, 0)


def rarity_reveal(model_id: str, rarity: str, newness: str, task: str,
                  backend: str, is_first: bool) -> None:
    rarity_colors = {
        "legendary": G,
        "epic":      PK,
        "rare":      T,
        "uncommon":  BL,
        "common":    DIM,
    }
    col = rarity_colors.get(rarity.lower(), T)
    stars = "★ ★ ★" if rarity == "legendary" else ("★ ★" if rarity == "epic" else "★")
    tags = []
    if is_first:
        tags.append(f"{G}★ FIRST TIME{RS}")
    if backend == "xla":
        tags.append(f"{CY}⚡ XLA{RS}")
    if newness == "zero_day":
        tags.append(f"{PK}◈ ZERO-DAY{RS}")
    elif newness == "hot":
        tags.append(f"{GN}◉ HOT{RS}")

    blank(1, 0)
    line(f"{col}{B}{'═' * 60}{RS}", 0.02)
    line(f"{col}{B}  {model_id[:54]}  {stars}{RS}", 0.03)
    line(f"{DIM}  task: {task}   rarity: {rarity.upper()}{'   ' + '   '.join(tags) if tags else ''}{RS}", 0.02)
    line(f"{col}{B}{'═' * 60}{RS}", 0.03)
    blank(1, 0)


def progress(step: int, total: int, msg: str, after: float = 0.04) -> None:
    line(f"  {DIM}▸ [{step}/{total}]{RS} {msg}", after)


def success_banner(model_id: str, compile_time: float, pts: int,
                   streak: int, is_first: bool, artifact: str = "") -> None:
    fire = f" 🔥×{streak}" if streak > 1 else ""
    first = f"  {G}★ FIRST!{RS}" if is_first else ""
    blank(1, 0)
    thick_sep(GN, 0.01)
    line(f"{B}{GN}  ✓  {model_id}  {compile_time:.1f}s  {B}+{pts} pts{RS}{fire}{first}", 0.05)
    if artifact:
        line(f"  {DIM}{artifact}{RS}", 0.03)
    thick_sep(GN, 0.04)
    blank(1, 0)


def failure_banner(model_id: str, error: str) -> None:
    blank(1, 0)
    sep("─", color=R)
    line(f"{R}{B}  ✗  {model_id}{RS}", 0.04)
    line(f"  {DIM}{error[:100]}{RS}", 0.03)
    sep("─", color=R, after=0.04)
    blank(1, 0)


def first_voice(text: str, sample: str) -> None:
    line(f"  {G}🗣  First Voice{RS}  {DIM}[{sample}]{RS}", 0.04)
    line(f"  {PK}{text}{RS}", 0.06)
    blank(1, 0)


# ════════════════════════════════════════════════════════════════════
# Intro / title card
# ════════════════════════════════════════════════════════════════════

emit("\033[2J\033[H")   # clear screen
pause(0.3)

figlet = r"""
  ████████╗████████╗      ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
     ██╔══╝╚══██╔══╝      ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
     ██║      ██║         █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
     ██║      ██║         ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
     ██║      ██║         ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
     ╚═╝      ╚═╝         ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝"""

for ln in figlet.splitlines():
    line(f"{B}{T}{ln}{RS}", 0.02)

blank(1, 0.0)
line(f"  {B}{G}C  O  M  P  I  L  E  T  R  O  N{RS}", 0.05)
blank(1, 0.0)
line(f"  {DIM}A roguelike model-compilation game for Tenstorrent Blackhole hardware{RS}", 0.05)
line(f"  {DIM}forge (PyTorch) + XLA/JAX (PJRT) dual-backend · live HuggingFace discovery{RS}", 0.05)
blank(2, 0.0)
pause(1.5)

# ════════════════════════════════════════════════════════════════════
# Setup / dispatch
# ════════════════════════════════════════════════════════════════════

sep("═", color=T, after=0.02)
line(f"{B}  EXPEDITION #007  ·  seed-only  ·  4 chips  ·  backend: auto{RS}", 0.03)
sep("═", color=T, after=0.04)
blank(1, 0)

line(f"  {DIM}Scanning tt-forge-models...{RS}", 0.05)
pause(0.3)
line(f"  {GN}✓{RS} {B}242 seed models{RS} found  {DIM}(11 ONNX filtered, 1 IRD-locked filtered){RS}", 0.05)
line(f"  {GN}✓{RS} {B}4 models queued{RS}  (1 per chip)", 0.05)
blank(1, 0)

dispatch = [
    ("0", "forge", "albert/token_classification/pytorch",  "nlp_token_cls"),
    ("1", "forge", "alexnet/pytorch",                       "cv_image_cls"),
    ("2", "forge", "allam/causal_lm/pytorch",               "nlp_causal_lm"),
    ("3", "xla  ", "albert/masked_lm/jax",                  "nlp_masked_lm"),
]

for cid, be, mid, task in dispatch:
    col = CY if "xla" in be else T
    line(f"  {col}⚡ Chip {cid} [{be}]{RS}  →  {B}{mid}{RS}  {DIM}{task}{RS}", 0.08)

blank(1, 0)
pause(0.8)

# ════════════════════════════════════════════════════════════════════
# CHIP 1 — alexnet/pytorch — fastest (2.6s)
# ════════════════════════════════════════════════════════════════════

chip_banner(1, "forge")
line(f"  1 model queued on TTDevice(id=1, arch=blackhole)\r\n", 0.04)

rarity_reveal("alexnet/pytorch", "rare", "hot", "CV Image Classification",
              "forge", is_first=True)

progress(1, 3, "Loading model via forge-models loader...", 0.06)
pause(0.4)
progress(2, 3, f"forge.compile()  on  TTDevice(id=1)...", 0.06)
pause(1.8)
progress(3, 3, f"Output: shape=(1, 1000) dtype=float32  range=[-3.99, 5.68]  (2.6s)", 0.06)
pause(0.2)

success_banner("alexnet/pytorch", 2.6, 300, streak=1, is_first=True,
               artifact="shape=(1, 1000)  range=[-3.99, 5.68]  ← ImageNet logits")
pause(0.5)

# ════════════════════════════════════════════════════════════════════
# CHIP 3 — albert/masked_lm/jax — XLA backend (8.2s)
# ════════════════════════════════════════════════════════════════════

chip_banner(3, "xla")
line(f"  1 model queued on TTDevice(id=3, arch=blackhole)  [JAX/PJRT]\r\n", 0.04)
line(f"  {CY}Initialising pjrt-plugin-tt 0.1.260319 ... {GN}✓{RS}  {DIM}(2.1s){RS}", 0.06)
blank(1, 0)

rarity_reveal("albert/masked_lm/jax", "epic", "zero_day", "NLP Masked LM",
              "xla", is_first=True)

progress(1, 3, "Loading Flax model  (_do_init=False, skipping eager XLA init)...", 0.06)
pause(0.5)
progress(2, 3, f"jax.jit → XLA compilation on  TTDevice(id=3, arch=blackhole)...", 0.06)
pause(4.8)
progress(3, 3, f"Output: shape=(1, 32, 30000)  (8.2s)", 0.06)
pause(0.2)

success_banner("albert/masked_lm/jax", 8.2, 600, streak=1, is_first=True,
               artifact="shape=(1, 32, 30000)  ← Flax BERT masked predictions")
pause(0.5)

# ════════════════════════════════════════════════════════════════════
# CHIP 0 — albert/token_classification/pytorch (9.2s)
# ════════════════════════════════════════════════════════════════════

chip_banner(0, "forge")
line(f"  1 model queued on TTDevice(id=0, arch=blackhole)\r\n", 0.04)

rarity_reveal("albert/token_classification/pytorch", "uncommon", "hot",
              "NLP Token Classification", "forge", is_first=True)

progress(1, 3, "Loading model via forge-models loader...", 0.06)
pause(0.4)
progress(2, 3, f"forge.compile()  on  TTDevice(id=0)...", 0.06)
pause(6.0)
progress(3, 3, f"Output: shape=(1, 32, 2) dtype=float32  (9.2s)", 0.06)
pause(0.2)

success_banner("albert/token_classification/pytorch", 9.2, 300, streak=1, is_first=True,
               artifact="shape=(1, 32, 2)  ← per-token logits (2 classes)")
pause(0.5)

# ════════════════════════════════════════════════════════════════════
# CHIP 2 — allam/causal_lm/pytorch — the long grind (55.8s)
# ════════════════════════════════════════════════════════════════════

chip_banner(2, "forge")
line(f"  1 model queued on TTDevice(id=2, arch=blackhole)\r\n", 0.04)

rarity_reveal("allam/causal_lm/pytorch", "legendary", "zero_day",
              "NLP Causal LM", "forge", is_first=True)

progress(1, 3, "Loading model via forge-models loader...", 0.06)
pause(1.2)
progress(2, 3, f"forge.compile()  on  TTDevice(id=2)...  {DIM}[large causal LM, patience]{RS}", 0.06)

# Show "still compiling" ticks for dramatic effect
for i in range(8):
    pause(3.5)
    line(f"  {DIM}  ··· {55.8 - (8-i)*3.5:.0f}s elapsed ···{RS}", 0.04)

pause(4.0)
progress(3, 3, f"Output: shape=(1, 32, 64000) dtype=float32  (55.8s)", 0.06)
pause(0.3)

success_banner("allam/causal_lm/pytorch", 55.8, 1500, streak=2, is_first=True,
               artifact="shape=(1, 32, 64000)  ← Allam vocab logits")
first_voice(
    "→ the (12%) | an (8%) | this (5%)",
    "At the Westinghouse pavilion, a time capsule was buried containing..."
)
pause(1.0)

# ════════════════════════════════════════════════════════════════════
# ALL CHIPS COMPLETE → countdown
# ════════════════════════════════════════════════════════════════════

blank(2, 0)
thick_sep(GN, 0.02)
line(f"{B}{GN}  ⚡ ALL CHIPS COMPLETE{RS}", 0.05)
for n in (3, 2, 1):
    pause(0.8)
    line(f"{DIM}  → Results in {n}...{RS}", 0.04)
thick_sep(GN, 0.05)
pause(1.0)

# ════════════════════════════════════════════════════════════════════
# SUMMARY SCREEN
# ════════════════════════════════════════════════════════════════════

emit("\033[2J\033[H")
pause(0.2)

thick_sep(G, 0.02)
line(f"{B}{G}  ⚡ EXPEDITION #007 COMPLETE   {DIM}2026-05-11{RS}", 0.04)
thick_sep(G, 0.05)
blank(1, 0)

# Run stats panel
line(f"  {B}THIS RUN{RS}", 0.03)
sep("─", 60, T, 0.02)
line(f"  4 attempted  ·  {GN}4 compiled{RS}  ·  0 failed  ·  {B}{GN}100%{RS}", 0.04)
line(f"  {T}{'█' * 36}{RS}  {B}+2,700 pts{RS}  {G}★ 4 new to bestiary{RS}", 0.04)
blank(1, 0)

# Chip leaderboard
line(f"  {B}CHIP LEADERBOARD{RS}", 0.03)
sep("─", 70, T, 0.02)

leaderboard = [
    ("🥇", 2, "forge", 1500, "✓1", 55.8, "★1"),
    ("🥈", 3, "xla  ", 600,  "✓1", 8.2,  "★1"),
    ("🥉", 0, "forge", 300,  "✓1", 9.2,  "★1"),
    ("   ", 1, "forge", 300,  "✓1", 2.6,  "★1"),
]

bar_max = 1500
for medal, cid, be, pts, wins, avg, news in leaderboard:
    col = CY if "xla" in be else T
    bar_len = int(36 * pts / bar_max)
    bar = f"{GN}{'█' * bar_len}{DIM}{'░' * (36 - bar_len)}{RS}"
    line(f"  {medal} C{cid} [{col}{be}{RS}]  {bar}  {B}+{pts:,}{RS}  {wins} {news}  {DIM}{avg:.1f}s avg{RS}", 0.06)

blank(1, 0)

# Compile time histogram
line(f"  {B}COMPILE TIMES{RS}  {DIM}(4 models){RS}", 0.03)
sep("─", 50, T, 0.02)
hist = [("< 5s",  1, 1), ("5–15s", 2, 2), ("15–60s", 1, 1), ("> 60s", 0, 0)]
for label, count, bar_n in hist:
    bar = f"{T}{'█' * (bar_n * 8)}{DIM}{'░' * ((3 - bar_n) * 8)}{RS}"
    line(f"  {label:8s}  {bar}  {count}", 0.05)

blank(1, 0)

# New to bestiary
line(f"  {G}{B}★ NEW TO BESTIARY (4){RS}", 0.04)
sep("─", 50, G, 0.02)
new_models = [
    ("alexnet/pytorch",                  "2.6s",  "+300"),
    ("albert/masked_lm/jax",             "8.2s",  "+600"),
    ("albert/token_classification",      "9.2s",  "+300"),
    ("allam/causal_lm/pytorch",          "55.8s", "+1,500 🗣"),
]
for mid, t_s, p in new_models:
    line(f"  {GN}✓{RS}  {B}{mid}{RS}  {DIM}{t_s}{RS}  {G}{p}{RS}", 0.06)

blank(1, 0)

# All-time stats
line(f"  {B}ALL-TIME BESTIARY{RS}", 0.03)
sep("─", 50, T, 0.02)
line(f"  {B}17{RS} models compiled  ·  best streak {G}🔥×3{RS}  ·  C2 leads {B}8,240 pts{RS}", 0.05)
blank(2, 0)

thick_sep(T, 0.02)
line(f"  {DIM}[R]{RS} Run Again   {DIM}[Q]{RS} Quit   {DIM}[B]{RS} Bestiary   {DIM}data/bestiary.json{RS}", 0.04)
thick_sep(T, 0.03)
blank(1, 0)
pause(5.0)

# ════════════════════════════════════════════════════════════════════
# Write cast to stdout
# ════════════════════════════════════════════════════════════════════

header = {
    "version": 2,
    "width": COLS,
    "height": ROWS,
    "timestamp": 1747000000,
    "title": "TT-Forge Compiletron — Expedition #007",
    "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
}

print(json.dumps(header))
for ev in events:
    print(json.dumps(ev))
