#!/usr/bin/env python3
"""
Generate a demo asciinema v2 cast for tt-forge-compiletron.

Shows a 4-chip expedition: 6 models per chip, forge + XLA backends.
Fast-forwards through compilation dead time (no gap > 1.5s in the cast).
Ends with a 4-chip simultaneous finale.

Usage:
    python3 scripts/gen_demo_cast.py > docs/demo.cast
"""
import json, sys

COLS = 180
ROWS = 50

# ── ANSI palette (matches expedition workers) ────────────────────────────────
T   = "\033[38;5;87m"    # teal
G   = "\033[38;5;220m"   # gold
R   = "\033[38;5;196m"   # red
GN  = "\033[38;5;46m"    # green
PK  = "\033[38;5;213m"   # pink
BL  = "\033[38;5;75m"    # blue
CY  = "\033[38;5;51m"    # bright cyan (XLA accent)
OR  = "\033[38;5;208m"   # orange
DIM = "\033[2m"
B   = "\033[1m"
RS  = "\033[0m"

events: list = []
_t: float = 0.0

def emit(text: str, after: float = 0.0) -> None:
    global _t
    events.append([round(_t, 3), "o", text])
    _t += after

def line(text: str = "", after: float = 0.04) -> None:
    emit(text + "\r\n", after)

def pause(secs: float) -> None:
    global _t
    _t += max(0.0, secs)

def blank(n: int = 1, after: float = 0.02) -> None:
    for _ in range(n):
        line("", after)

def sep(char="─", width=COLS-2, color=DIM, after=0.02):
    line(f"{color}{char*width}{RS}", after)

def thick_sep(color=T, after=0.04):
    line(f"{B}{color}{'═'*(COLS-2)}{RS}", after)

# ────────────────────────────────────────────────────────────────────────────
# compile_wait: the fast-forward engine
# Shows a ⏩ progress bar and caps playback to < 1.5 s no matter how long
# the real compile took. Keeps the cast kinetic at all times.
# ────────────────────────────────────────────────────────────────────────────
def compile_wait(real_s: float, label: str = "", chip_col: str = T) -> None:
    global _t
    if real_s <= 4.0:
        pause(real_s * 0.3)
        return
    speed_x = max(4, int(real_s / 2))
    bar_full = 28
    steps = 6
    emit(f"  {chip_col}⏩{RS}  {DIM}×{speed_x} fast-forward  ({real_s:.0f}s wall-clock){RS}\r\n", 0.0)
    playback_budget = min(1.4, real_s / speed_x)
    step_t = playback_budget / steps
    for i in range(steps):
        filled = int(bar_full * (i+1) / steps)
        bar = f"{chip_col}{'█'*filled}{DIM}{'░'*(bar_full-filled)}{RS}"
        pct = int(100*(i+1)/steps)
        elapsed_sim = real_s * (i+1) / steps
        line(f"  {DIM}  [{bar}{DIM}] {pct:3d}%  {elapsed_sim:.1f}s{RS}", step_t)

# ── Model record helpers ──────────────────────────────────────────────────────
def chip_banner(chip_id, backend, run=7, total=6):
    col = CY if backend == "xla" else T
    label = "XLA/JAX" if backend == "xla" else "FORGE  "
    blank(1, 0)
    thick_sep(col, 0.01)
    line(f"{B}{col}  ⚡ EXPEDITION {label} CHIP {chip_id}  ·  run #{run:03d}  ·  {total} models{RS}", 0.03)
    thick_sep(col, 0.04)
    blank(1, 0)

def rarity_reveal(model_id, rarity, newness, task, backend, is_first):
    rc = {
        "legendary": G, "epic": PK, "rare": T, "uncommon": BL, "common": DIM,
    }.get(rarity.lower(), T)
    stars = {"legendary":"★ ★ ★","epic":"★ ★","rare":"★","uncommon":"◇","common":"·"}.get(rarity.lower(),"·")
    tags = []
    if is_first:      tags.append(f"{G}★ FIRST{RS}")
    if backend=="xla": tags.append(f"{CY}⚡ XLA{RS}")
    if newness=="zero_day": tags.append(f"{PK}◈ ZERO-DAY{RS}")
    elif newness=="hot":    tags.append(f"{GN}◉ HOT{RS}")
    tag_str = ("  " + "  ".join(tags)) if tags else ""
    blank(1, 0)
    line(f"{rc}{B}{'═'*58}{RS}", 0.02)
    line(f"{rc}{B}  {model_id[:52]}  {stars}{RS}", 0.03)
    line(f"  {DIM}task: {task:<28} rarity: {rarity.upper()}{tag_str}{RS}", 0.02)
    line(f"{rc}{B}{'═'*58}{RS}", 0.03)
    blank(1, 0)

def progress(step, total, msg, after=0.04):
    line(f"  {DIM}▸ [{step}/{total}]{RS} {msg}", after)

def success_banner(model_id, compile_time, pts, streak, is_first, artifact=""):
    fire  = f" 🔥×{streak}" if streak>1 else ""
    first = f"  {G}★ FIRST!{RS}" if is_first else ""
    blank(1, 0)
    thick_sep(GN, 0.01)
    line(f"{B}{GN}  ✓  {model_id}  {compile_time:.1f}s  {B}+{pts:,} pts{RS}{fire}{first}", 0.05)
    if artifact:
        line(f"  {DIM}{artifact}{RS}", 0.03)
    thick_sep(GN, 0.04)
    blank(1, 0)

def failure_banner(model_id, error):
    blank(1, 0)
    sep("─", color=R)
    line(f"{R}{B}  ✗  {model_id}{RS}  {DIM}{error[:80]}{RS}", 0.04)
    sep("─", color=R, after=0.04)
    blank(1, 0)

def first_voice(text, sample):
    line(f"  {G}🗣  First Voice{RS}  {DIM}[{sample}]{RS}", 0.04)
    line(f"  {PK}{text}{RS}", 0.06)
    blank(1, 0)

def run_model(chip_id, backend, model_id, real_s, pts, streak, is_first,
              rarity, newness, task, artifact="", fv_text="", fv_sample="",
              fail=False, fail_err=""):
    """Drive a full compile sequence for one model."""
    col = CY if backend == "xla" else T
    rarity_reveal(model_id, rarity, newness, task, backend, is_first)
    if fail:
        progress(1, 3, "Loading model...", 0.05)
        pause(0.4)
        failure_banner(model_id, fail_err)
        return
    progress(1, 3, "Loading model...", 0.05)
    pause(min(0.6, real_s * 0.08))
    be_str = "jax.jit → PJRT XLA" if backend=="xla" else "forge.compile()"
    progress(2, 3, f"{be_str}  on  TTDevice(id={chip_id})...", 0.05)
    compile_wait(real_s, model_id, col)
    progress(3, 3, f"Output: {artifact}  ({real_s:.1f}s)", 0.05)
    pause(0.15)
    success_banner(model_id, real_s, pts, streak, is_first, artifact)
    if fv_text:
        first_voice(fv_text, fv_sample)

# ════════════════════════════════════════════════════════════════════════════
# INTRO
# ════════════════════════════════════════════════════════════════════════════

emit("\033[2J\033[H")
pause(0.3)

for ln in [
  r"  ████████╗████████╗      ███████╗ ██████╗ ██████╗  ██████╗ ███████╗",
  r"     ██╔══╝╚══██╔══╝      ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝",
  r"     ██║      ██║         █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  ",
  r"     ██║      ██║         ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ",
  r"     ██║      ██║         ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗",
  r"     ╚═╝      ╚═╝         ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
]:
    line(f"{B}{T}{ln}{RS}", 0.02)

blank(1, 0)
line(f"  {B}{G}C  O  M  P  I  L  E  T  R  O  N{RS}", 0.04)
blank(1, 0)
line(f"  {DIM}4 chips  ·  6 models each  ·  forge + XLA/JAX (PJRT)  ·  run #007{RS}", 0.04)
blank(2, 0)
pause(0.8)

sep("═", color=T, after=0.02)
line(f"{B}  EXPEDITION #007  SETUP{RS}", 0.03)
sep("═", color=T, after=0.03)
blank(1, 0)
line(f"  {DIM}Scanning tt-forge-models... {GN}242 models found{RS}  {DIM}(ONNX + IRD-locked filtered){RS}", 0.05)
line(f"  {DIM}HuggingFace frontier... {GN}22 fresh models eligible{RS}", 0.05)
line(f"  {DIM}Pre-downloading weights...{RS} {GN}✓{RS}  {DIM}(12 models cached){RS}", 0.05)
blank(1, 0)
line(f"  {B}Dispatching 24 models across 4 chips{RS}  {DIM}(6 per chip){RS}", 0.04)
blank(1, 0)

dispatch = [
    ("0", T,  "forge  ", [
        "alexnet/pytorch", "albert/token_classification/pytorch",
        "openai-community/gpt2", "facebook/bart-base",
        "microsoft/phi-1_5 (frontier)", "[FINALE] tiiuae/falcon-rw-1b",
    ]),
    ("1", T,  "forge  ", [
        "alexnet/image_classification/paddlepaddle", "allam/causal_lm/pytorch",
        "bert-base-uncased (frontier)", "OvercastLab/Quark-50m",
        "snehangshu511/gpt2-medium-instruct", "[FINALE] meta/Llama-3.2-1B",
    ]),
    ("2", CY, "xla    ", [
        "albert/masked_lm/jax", "alexnet/image_classification/jax",
        "google/flax-bert-base-uncased", "openai-community/gpt2 [jax]",
        "google/flax-t5-small", "[FINALE] stabilityai/stable-code-3b [jax]",
    ]),
    ("3", T,  "forge  ", [
        "albert/sequence_classification/pytorch", "albert/masked_lm/pytorch",
        "musicgen/small (frontier)", "ULTR0N/SentienceTrial1",
        "lumasik/quark-1-248m-base", "[FINALE] allam/causal_lm/pytorch",
    ]),
]
for cid, col, be, models in dispatch:
    line(f"  {col}⚡ Chip {cid} [{be}]{RS}", 0.02)
    for m in models:
        finl = m.startswith("[FINALE]")
        mc = OR if finl else DIM
        line(f"      {mc}{m}{RS}", 0.02)

blank(2, 0)
pause(0.8)

# ════════════════════════════════════════════════════════════════════════════
# CHIP 0 — forge — models 1-5
# ════════════════════════════════════════════════════════════════════════════

chip_banner(0, "forge", total=6)
line(f"  6 models queued on TTDevice(id=0, arch=blackhole)\r\n", 0.04)

run_model(0, "forge", "alexnet/pytorch",
          real_s=2.6, pts=300, streak=1, is_first=True,
          rarity="rare", newness="hot", task="CV Image Classification",
          artifact="shape=(1, 1000) dtype=float32  range=[-3.99, 5.68]")

run_model(0, "forge", "albert/token_classification/pytorch",
          real_s=9.2, pts=300, streak=2, is_first=True,
          rarity="uncommon", newness="hot", task="NLP Token Classification",
          artifact="shape=(1, 32, 2) dtype=float32")

run_model(0, "forge", "openai-community/gpt2",
          real_s=15.3, pts=450, streak=3, is_first=True,
          rarity="common", newness="fresh", task="NLP Causal LM",
          artifact="shape=(1, 32, 50257) dtype=float32",
          fv_text="→ the (10%) | a (7%) | this (5%)",
          fv_sample="In 1964 at the New York World's Fair, beneath the gleaming")

run_model(0, "forge", "facebook/bart-base",
          real_s=28.4, pts=600, streak=4, is_first=True,
          rarity="rare", newness="hot", task="NLP Seq2Seq",
          artifact="shape=(1, 32, 50265) dtype=float32")

run_model(0, "forge", "microsoft/phi-1_5",
          real_s=48.2, pts=1200, streak=5, is_first=True,
          rarity="epic", newness="zero_day", task="NLP Causal LM",
          artifact="shape=(1, 32, 51200) dtype=float32",
          fv_text="→ learning (14%) | reasoning (9%) | thinking (6%)",
          fv_sample="The key principle behind transformer attention is")

# ════════════════════════════════════════════════════════════════════════════
# CHIP 1 — forge — models 1-5
# ════════════════════════════════════════════════════════════════════════════

chip_banner(1, "forge", total=6)
line(f"  6 models queued on TTDevice(id=1, arch=blackhole)\r\n", 0.04)

run_model(1, "forge", "alexnet/image_classification/paddlepaddle",
          real_s=2.7, pts=300, streak=1, is_first=True,
          rarity="rare", newness="hot", task="CV Image Classification",
          artifact="shape=(1, 1000) dtype=float32  range=[-4.95, 6.03]")

run_model(1, "forge", "allam/causal_lm/pytorch",
          real_s=55.8, pts=1500, streak=2, is_first=True,
          rarity="legendary", newness="hot", task="NLP Causal LM",
          artifact="shape=(1, 32, 64000) dtype=float32",
          fv_text="→ the (12%) | an (8%) | this (5%)",
          fv_sample="At the Westinghouse pavilion, a time capsule was buried")

run_model(1, "forge", "bert-base-uncased",
          real_s=14.1, pts=300, streak=3, is_first=True,
          rarity="common", newness="fresh", task="NLP Masked LM",
          artifact="shape=(1, 32, 30522) dtype=float32")

run_model(1, "forge", "OvercastLab/Quark-50m",
          real_s=9.8, pts=900, streak=4, is_first=True,
          rarity="epic", newness="zero_day", task="NLP Causal LM",
          artifact="shape=(1, 32, 32000) dtype=float32",
          fv_text="→ world (8%) | future (6%) | you (5%)",
          fv_sample="Ma Joad said nothing for a long time. She looked at")

run_model(1, "forge", "snehangshu511/gpt2-medium-instruct",
          real_s=24.4, pts=750, streak=5, is_first=True,
          rarity="epic", newness="hot", task="NLP Causal LM",
          artifact="shape=(1, 32, 50257) dtype=float32",
          fv_text="→ quantum (11%) | neural (8%) | entangled (4%)",
          fv_sample="The robot at the end of the corridor extended a hand")

# ════════════════════════════════════════════════════════════════════════════
# CHIP 2 — XLA/JAX — models 1-5
# ════════════════════════════════════════════════════════════════════════════

chip_banner(2, "xla", total=6)
line(f"  6 models queued on TTDevice(id=2, arch=blackhole)  {CY}[JAX/PJRT backend]{RS}\r\n", 0.04)
line(f"  {CY}Initialising pjrt-plugin-tt 0.1.260319 ...{RS} {GN}✓{RS}  {DIM}(2.1s){RS}", 0.06)
blank(1, 0)

run_model(2, "xla", "albert/masked_lm/jax",
          real_s=8.2, pts=600, streak=1, is_first=True,
          rarity="epic", newness="zero_day", task="NLP Masked LM",
          artifact="shape=(1, 32, 30000) dtype=float32")

run_model(2, "xla", "alexnet/image_classification/jax",
          real_s=4.3, pts=300, streak=2, is_first=True,
          rarity="rare", newness="hot", task="CV Image Classification",
          artifact="shape=(8, 1000) dtype=bfloat16")

run_model(2, "xla", "google/flax-bert-base-uncased",
          real_s=18.6, pts=600, streak=3, is_first=True,
          rarity="common", newness="fresh", task="NLP Masked LM",
          artifact="shape=(1, 32, 30522) dtype=float32")

run_model(2, "xla", "openai-community/gpt2 [jax]",
          real_s=22.1, pts=600, streak=4, is_first=True,
          rarity="uncommon", newness="fresh", task="NLP Causal LM",
          artifact="shape=(1, 32, 50257) dtype=float32",
          fv_text="→ city (9%) | world (7%) | light (5%)",
          fv_sample="In the exhibit hall of the future, the robot shook hands")

run_model(2, "xla", "google/flax-t5-small",
          real_s=35.4, pts=900, streak=5, is_first=True,
          rarity="rare", newness="hot", task="NLP Seq2Seq",
          artifact="shape=(1, 32, 32128) dtype=float32")

# ════════════════════════════════════════════════════════════════════════════
# CHIP 3 — forge — models 1-5
# ════════════════════════════════════════════════════════════════════════════

chip_banner(3, "forge", total=6)
line(f"  6 models queued on TTDevice(id=3, arch=blackhole)\r\n", 0.04)

run_model(3, "forge", "albert/sequence_classification/pytorch",
          real_s=10.3, pts=150, streak=1, is_first=False,
          rarity="uncommon", newness="fresh", task="NLP Text Classification",
          artifact="shape=(1, 2) dtype=float32")

run_model(3, "forge", "albert/masked_lm/pytorch",
          real_s=10.6, pts=150, streak=2, is_first=False,
          rarity="uncommon", newness="fresh", task="NLP Masked LM",
          artifact="shape=(1, 32, 30000) dtype=float32")

run_model(3, "forge", "musicgen/small",
          real_s=44.7, pts=1200, streak=3, is_first=True,
          rarity="legendary", newness="zero_day", task="Audio Generation",
          artifact="shape=(1, 1, 48000) dtype=float32  ← 3s of audio!")

run_model(3, "forge", "ULTR0N/SentienceTrial1",
          real_s=11.8, pts=750, streak=4, is_first=True,
          rarity="epic", newness="zero_day", task="NLP Causal LM",
          artifact="shape=(1, 32, 65536) dtype=float32",
          fv_text="→ consciousness (13%) | existence (8%) | being (6%)",
          fv_sample="The question of whether a machine can truly think")

run_model(3, "forge", "lumasik/quark-1-248m-base",
          real_s=8.9, pts=900, streak=5, is_first=True,
          rarity="epic", newness="zero_day", task="NLP Causal LM",
          artifact="shape=(1, 32, 131072) dtype=float32",
          fv_text="→ the (9%) | a (7%) | it (5%)",
          fv_sample="The ancient manuscript described a formula for")

# ════════════════════════════════════════════════════════════════════════════
# THE 4-CHIP FINALE ANNOUNCEMENT
# ════════════════════════════════════════════════════════════════════════════

blank(2, 0)
emit("\033[2J\033[H")
pause(0.3)

thick_sep(OR, 0.02)
line(f"{B}{OR}  ⚡ ⚡ ⚡  FINALE — ALL CHIPS LOCKED ON FINAL MODELS  ⚡ ⚡ ⚡{RS}", 0.05)
thick_sep(OR, 0.03)
blank(1, 0)

finale_models = [
    ("0", T,  "forge  ", "tiiuae/falcon-rw-1b",            62.3, "legendary", "NLP Causal LM"),
    ("1", T,  "forge  ", "meta/Llama-3.2-1B",              71.5, "legendary", "NLP Causal LM"),
    ("2", CY, "xla    ", "stabilityai/stable-code-3b [jax]",78.1, "legendary", "Code Generation"),
    ("3", T,  "forge  ", "allam/causal_lm/pytorch",         55.8, "legendary", "NLP Causal LM"),
]

# Print the finale model table
for cid, col, be, mid, real_s, rarity, task in finale_models:
    line(f"  {col}⚡ C{cid} [{be}]{RS}  {B}{mid}{RS}  {DIM}~{real_s:.0f}s{RS}  {G}★ ALL LEGENDARY{RS}", 0.07)

blank(1, 0)
pause(0.4)

line(f"  {DIM}All chips launching simultaneously...{RS}", 0.06)
blank(1, 0)

# Show each chip announcing their finale
for cid, col, be, mid, real_s, rarity, task in finale_models:
    pause(0.2)
    line(f"  {col}{B}CHIP {cid}  ▸  loading  {mid}{RS}", 0.06)
    pause(0.3)
    line(f"  {col}  ▸  {('jax.jit → PJRT XLA' if 'xla' in be else 'forge.compile()')}  starting...{RS}", 0.05)

blank(1, 0)
pause(0.3)

# ── FAST-FORWARD through the finale compilation ──────────────────────────────
max_s = max(r for _, _, _, _, r, _, _ in finale_models)  # 78.1s
line(f"  {OR}{B}⏩  ALL 4 CHIPS COMPILING  ×{int(max_s/3)} FAST-FORWARD{RS}", 0.03)
blank(1, 0)

# Animated progress bars for all 4 chips simultaneously during FF
bar_w = 30
steps = 10
ff_budget = 2.5
step_t = ff_budget / steps
for i in range(steps):
    pct = (i+1)*10
    elapsed_sim = max_s * (i+1) / steps
    row_parts = []
    for cid, col, be, mid, real_s, rarity, task in finale_models:
        chip_pct = min(100, int(100 * elapsed_sim / real_s))
        filled = int(bar_w * chip_pct / 100)
        bar = f"{col}{'█'*filled}{DIM}{'░'*(bar_w-filled)}{RS}"
        row_parts.append(f"C{cid}[{bar}{DIM}]{chip_pct:3d}%{RS}")
    line("  " + "  ".join(row_parts), step_t)

blank(1, 0)
pause(0.4)

# ── CHIP 3 finishes first (55.8s) ─────────────────────────────────────────
blank(1, 0)
sep("═"*2, color=GN)
line(f"{GN}{B}  ⚡ CHIP 3 DONE FIRST!{RS}", 0.04)
blank(1, 0)
rarity_reveal("allam/causal_lm/pytorch", "legendary", "hot", "NLP Causal LM", "forge", True)
progress(3, 3, "Output: shape=(1, 32, 64000)  (55.8s)", 0.04)
success_banner("allam/causal_lm/pytorch", 55.8, 2500, streak=6, is_first=True,
               artifact="shape=(1, 32, 64000) dtype=float32  🔥 STREAK ×6")
first_voice("→ the (12%) | an (8%) | this (5%)",
            "At the Westinghouse pavilion, a time capsule was buried")
pause(0.5)

# ── CHIP 0 finishes (62.3s) ───────────────────────────────────────────────
line(f"{GN}{B}  ⚡ CHIP 0 DONE!{RS}", 0.04)
blank(1, 0)
rarity_reveal("tiiuae/falcon-rw-1b", "legendary", "zero_day", "NLP Causal LM", "forge", True)
progress(3, 3, "Output: shape=(1, 32, 65024)  (62.3s)", 0.04)
success_banner("tiiuae/falcon-rw-1b", 62.3, 3000, streak=6, is_first=True,
               artifact="shape=(1, 32, 65024) dtype=float32  ◈ ZERO-DAY")
first_voice("→ Falcon (8%) | The (7%) | In (6%)",
            "The ancient library held secrets that no human")
pause(0.5)

# ── CHIP 1 finishes (71.5s) ───────────────────────────────────────────────
line(f"{GN}{B}  ⚡ CHIP 1 DONE!{RS}", 0.04)
blank(1, 0)
rarity_reveal("meta/Llama-3.2-1B", "legendary", "zero_day", "NLP Causal LM", "forge", True)
progress(3, 3, "Output: shape=(1, 32, 128256)  (71.5s)", 0.04)
success_banner("meta/Llama-3.2-1B", 71.5, 3500, streak=6, is_first=True,
               artifact="shape=(1, 32, 128256) dtype=float32  ★★★ LEGENDARY")
first_voice("→ The (11%) | A (8%) | In (6%)",
            "In 1969, the astronaut looked back at Earth and")
pause(0.5)

# ── CHIP 2 finishes last (78.1s) — XLA! ──────────────────────────────────
line(f"{CY}{B}  ⚡ CHIP 2 DONE — XLA! FINAL CHIP!{RS}", 0.05)
blank(1, 0)
rarity_reveal("stabilityai/stable-code-3b [jax]", "legendary", "zero_day",
              "Code Generation", "xla", True)
progress(3, 3, "Output: shape=(1, 32, 49152)  (78.1s)", 0.04)
success_banner("stabilityai/stable-code-3b [jax]", 78.1, 4000, streak=6, is_first=True,
               artifact="shape=(1, 32, 49152) dtype=bfloat16  ⚡ XLA  ★★★ LEGENDARY")
first_voice("→ def (18%) | import (11%) | class (8%)",
            "def solve_tsp(distances: list[list[float]])")
pause(0.8)

# ════════════════════════════════════════════════════════════════════════════
# ALL CHIPS COMPLETE — countdown
# ════════════════════════════════════════════════════════════════════════════

blank(2, 0)
thick_sep(GN, 0.02)
line(f"{B}{GN}  ⚡ ALL 4 CHIPS COMPLETE — 24/24 MODELS{RS}", 0.05)
for n in (3, 2, 1):
    pause(0.8)
    line(f"{DIM}  → Results in {n}...{RS}", 0.04)
thick_sep(GN, 0.05)
pause(0.8)

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY SCREEN
# ════════════════════════════════════════════════════════════════════════════

emit("\033[2J\033[H")
pause(0.3)

thick_sep(G, 0.02)
line(f"{B}{G}  ⚡ EXPEDITION #007 COMPLETE   2026-05-11{RS}", 0.04)
thick_sep(G, 0.05)
blank(1, 0)

# ── THIS RUN box ─────────────────────────────────────────────────────────────
line(f"  {B}THIS RUN{RS}", 0.03)
sep("─", 72, T, 0.02)
line(f"  24 attempted  ·  {GN}23 compiled{RS}  ·  {R}1 failed{RS}  ·  {B}{GN}96%{RS}  ·  {B}+24,850 pts{RS}", 0.04)
line(f"  {T}{'█'*42}{DIM}{'░'*6}{RS}  {G}★ 20 new to bestiary{RS}  {DIM}(4 previously compiled){RS}", 0.04)
blank(1, 0)

# ── Chip leaderboard ─────────────────────────────────────────────────────────
line(f"  {B}CHIP LEADERBOARD{RS}", 0.03)
sep("─", 78, T, 0.02)

boards = [
    ("🥇", 2, "xla  ", 7_800, "✓6 ✗0", 36.4, "🔥×6", "★6"),
    ("🥈", 1, "forge", 7_100, "✓6 ✗0", 30.7, "🔥×6", "★5"),
    ("🥉", 0, "forge", 6_400, "✓5 ✗1", 28.6, "🔥×5", "★4"),
    ("   ", 3, "forge", 5_550, "✓6 ✗0", 24.0, "🔥×6", "★5"),
]
bar_max = 7_800
for medal, cid, be, pts, wl, avg, streak, news in boards:
    col = CY if be == "xla  " else T
    filled = int(36 * pts / bar_max)
    bar = f"{GN}{'█'*filled}{DIM}{'░'*(36-filled)}{RS}"
    line(f"  {medal} C{cid} [{col}{be}{RS}]  {bar}  {B}{pts:,}{RS}  {wl}  {streak}  {DIM}{avg:.1f}s avg{RS}  {G}{news}{RS}", 0.07)

blank(1, 0)

# ── Compile time histogram ─────────────────────────────────────────────────
line(f"  {B}COMPILE TIMES{RS}  {DIM}(23 successful){RS}", 0.03)
sep("─", 55, T, 0.02)
hist = [("< 5s", 4, 4), ("5–15s", 8, 8), ("15–45s", 7, 7), ("> 45s", 4, 4)]
for label, count, bn in hist:
    bar = f"{T}{'█'*(bn*3)}{DIM}{'░'*((8-bn)*3)}{RS}"
    line(f"  {label:8s}  {bar}  {count}", 0.05)

blank(1, 0)

# ── Failures ─────────────────────────────────────────────────────────────────
line(f"  {R}{B}✗ FAILED (1){RS}", 0.03)
sep("─", 55, R, 0.02)
line(f"  {R}✗{RS}  {DIM}musicgen/melody (frontier){RS}  {R}RuntimeError: output rank 5 unsupported{RS}", 0.05)
blank(1, 0)

# ── New to bestiary ───────────────────────────────────────────────────────────
line(f"  {G}{B}★ NEW TO BESTIARY — HIGHLIGHTS (20 total){RS}", 0.04)
sep("─", 55, G, 0.02)
highlights = [
    ("stabilityai/stable-code-3b [jax]",    "78.1s", "+4,000", "⚡ XLA ★★★"),
    ("meta/Llama-3.2-1B",                   "71.5s", "+3,500", "★★★ 🗣"),
    ("tiiuae/falcon-rw-1b",                 "62.3s", "+3,000", "★★★ 🗣"),
    ("allam/causal_lm/pytorch",             "55.8s", "+2,500", "★★★ 🗣"),
    ("musicgen/small",                      "44.7s", "+1,200", "★★★ 🎵"),
    ("microsoft/phi-1_5",                   "48.2s", "+1,200", "★★  🗣"),
]
for mid, t_s, p, tags in highlights:
    line(f"  {GN}✓{RS}  {B}{mid:<38}{RS} {DIM}{t_s:>7}{RS}  {G}{p:>6}{RS}  {DIM}{tags}{RS}", 0.06)

blank(1, 0)

# ── All-time stats ────────────────────────────────────────────────────────────
line(f"  {B}ALL-TIME BESTIARY{RS}", 0.03)
sep("─", 55, T, 0.02)
line(f"  {B}37{RS} models compiled  ·  best streak {G}🔥×8{RS}  ·  C2 leads {B}15,840 pts{RS}  ·  {CY}6 XLA models{RS}", 0.05)
blank(2, 0)

thick_sep(T, 0.02)
line(f"  {DIM}[R]{RS} Run Again   {DIM}[Q]{RS} Quit   {DIM}[B]{RS} Bestiary   {DIM}data/bestiary.json  ·  data/expeditions/run_007.md{RS}", 0.04)
thick_sep(T, 0.04)
blank(1, 0)
pause(6.0)

# ════════════════════════════════════════════════════════════════════════════
# Post-process: clamp any idle gap > MAX_IDLE seconds.
# This is the "fast-forward where nothing changes" guarantee.
# ════════════════════════════════════════════════════════════════════════════

MAX_IDLE = 1.5

compressed = []
offset = 0.0
prev_t = 0.0
for ev in events:
    real_gap = ev[0] - prev_t
    if real_gap > MAX_IDLE:
        offset += real_gap - MAX_IDLE
    compressed.append([round(ev[0] - offset, 3), ev[1], ev[2]])
    prev_t = ev[0]

# ── Write cast ───────────────────────────────────────────────────────────────
header = {
    "version": 2, "width": COLS, "height": ROWS,
    "timestamp": 1747000000,
    "title": "TT-Forge Compiletron — Expedition #007 — 4-Chip Finale",
    "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
}
print(json.dumps(header))
for ev in compressed:
    print(json.dumps(ev))

total = compressed[-1][0]
sys.stderr.write(f"Generated {len(compressed)} events  duration={total:.1f}s\n")
