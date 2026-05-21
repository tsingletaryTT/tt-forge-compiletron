#!/usr/bin/env python3
# lib/expedition/expedition_worker.py
"""
Per-chip expedition worker. Runs the forge compile pipeline for each model
in this chip's queue, then pipes results through decoder → scorer → hud.

Invoked by run_expedition.sh as:
  python3 lib/expedition/expedition_worker.py \
      --chip N --run R --bestiary data/bestiary.json \
      --queue /tmp/expedition_queue_chipN.json \
      --results /tmp/expedition_results_chipN.csv
"""
from __future__ import annotations

# Ensure the project root is in sys.path regardless of how this script is
# invoked (python3 /abs/path/expedition_worker.py puts the script's own
# directory first, not the repo root — breaking all lib.* imports).
import sys as _sys
from pathlib import Path as _Path
_project_root = str(_Path(__file__).resolve().parent.parent.parent)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import csv
import json
import multiprocessing
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

# Silence C++ noise before any TT imports — same technique as lib/worker.py
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")

import warnings
warnings.filterwarnings("ignore")

# ── Forge / TT Metal pre-load ─────────────────────────────────────────────────
# Tell forge's module.py and tvm_utils.py to skip TensorFlow / JAX / Paddle
# imports.  Those frameworks ship with their own LLVM builds; when loaded in
# the same process as libTTMLIRCompiler.so (which exports ALL LLVM symbols via
# --whole-archive), TF's global C++ constructors call into our CommandLineParser
# singleton in an incompatible state, triggering a SmallPtrSet assertion and
# SIGSEGV.  Skipping those imports fixes the crash with no functional loss for
# the PyTorch-only compilation pipeline used here.
os.environ["FORGE_PYTORCH_ONLY"] = "1"

# forge._C (the PyTorch extension) must be loaded BEFORE TensorFlow or any
# other GPU library, because TF's global C++ constructors allocate GPU memory
# that interferes with TT Metal's device init when they share the same process.
# Pre-loading the MLIR/Metal shared objects via ctypes first ensures they win
# the dlopen init race, then importing forge locks in the correct state.
# This must happen at module level — deferring it to _compile_model() is too
# late because model loader imports (torch → TF) fire during _build_loader().
try:
    import ctypes as _ctypes, os as _os
    _install_lib = _os.path.join(
        _os.path.expanduser("~/tt-forge-fe"),
        "third_party/tt-mlir/build/install/lib",
    )
    for _lib in [
        "libdevice.so", "libtt_metal.so",
        "libTTMLIRRuntime.so", "libTTMLIRCompiler.so", "libTTNNCompileSo.so",
    ]:
        _p = _os.path.join(_install_lib, _lib)
        if _os.path.exists(_p):
            _ctypes.CDLL(_p)
    import sys as _sys2
    _sys2.path.insert(0, _os.path.expanduser("~/tt-forge-fe"))
    import forge as _forge_preload  # noqa: F401  — side-effect: initialises TT Metal
    del _sys2, _ctypes, _os, _forge_preload
except Exception:
    pass  # non-forge runs (XLA, ONNX) don't need forge loaded at startup


# ── ANSI colors (Tenstorrent palette) ────────────────────────────────────────

BOLD   = "\033[1m"
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
PURPLE = "\033[35m"
PINK   = "\033[95m"
BLUE   = "\033[94m"
GOLD   = "\033[33m"
DIM    = "\033[2m"


# ── Rarity display config ────────────────────────────────────────────────────

# Maps rarity tier → (ANSI color code, display label, pause seconds).
# Pause causes the terminal to linger briefly on special reveals so the user
# has time to appreciate the banner before compilation output floods in.
_RARITY_STYLE = {
    "legendary": (PURPLE, "★★★ LEGENDARY", 2.0),
    "rare":      (PINK,   "★ RARE FIND",   1.0),
    "uncommon":  (YELLOW, "◆ UNCOMMON",     0.5),
    "common":    (CYAN,   "",               0.0),
    "familiar":  (CYAN,   "",               0.0),
}

# Maps newness tier → (ANSI color code, display label, pause seconds).
_NEWNESS_STYLE = {
    "zero_day":    (GOLD,   "⚡ ZERO DAY",  3.0),
    "hot":         (YELLOW, "🔥 HOT",        0.5),
    "fresh":       (GREEN,  "✨ FRESH",       0.0),
    "established": ("",     "",              0.0),
    "familiar":    ("",     "",              0.0),
}

# Fixed fieldnames for the per-chip CSV results file.
# Using a module-level constant (rather than results[0].keys()) prevents
# ValueError when success rows and failure rows have different key sets.
# extrasaction="ignore" in DictWriter allows both row shapes to coexist.
_CSV_FIELDNAMES = [
    "model", "status", "pts", "compile_time", "infer_time",
    "artifact", "first_ever", "first_voice", "error",
]

# Tasks whose output tensor's seq dimension (axis 1) is meaningful as token
# count.  For these tasks _compute_throughput reports tokens/sec; everything
# else (CV, embeddings, QA, audio) reports ms/sample.
_TOKEN_TASKS: frozenset[str] = frozenset({
    "text-generation", "nlp_causal_lm", "nlp_masked_lm",
    "fill-mask", "nlp_text_cls", "nlp_token_cls",
})


def _compute_throughput(task: str, output: Any, infer_s: float) -> tuple[float, str]:
    """Compute throughput from a decoded inference output tensor.

    For token-producing tasks (members of _TOKEN_TASKS), returns
    (tokens_per_sec, "tokens/sec") using output.shape[1] as the sequence
    length.  For all other tasks (CV, embeddings, QA, audio) returns
    (ms_per_sample, "ms/sample").  Returns (0.0, "") when infer_s is zero,
    output is None, or the output shape cannot be read (e.g. 1-D tensor for
    a token task).

    Args:
        task:    HuggingFace pipeline task string (e.g. "text-generation").
        output:  Raw inference output tensor (may be None on failure paths).
        infer_s: Wall-clock seconds spent in the inference call.  Must be > 0.

    Returns:
        (throughput_value, throughput_unit) where unit is "tokens/sec",
        "ms/sample", or "" (when measurement is not available).
    """
    if infer_s <= 0.0 or output is None:
        return 0.0, ""
    try:
        if task in _TOKEN_TASKS:
            seq_len = output.shape[1]
            return round(seq_len / infer_s, 3), "tokens/sec"
    except (AttributeError, IndexError):
        # output has no .shape or shape has fewer than 2 dimensions — fall
        # through to the ms/sample path rather than crashing.
        pass
    return round(infer_s * 1000.0, 3), "ms/sample"


def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) from a sorted list via linear interpolation.

    Uses the same formula as numpy.percentile with interpolation='linear':
    locate the fractional index, then linearly interpolate between the two
    neighbouring values.  Works correctly for edge cases:
      - Single element: returns that element regardless of p.
      - p=0: returns sorted_data[0].
      - p=100: returns sorted_data[-1].

    Args:
        sorted_data: Pre-sorted (ascending) list of float values.
        p:           Percentile in the range [0, 100].

    Returns:
        Interpolated percentile value, or 0.0 for an empty list.
    """
    if not sorted_data:
        return 0.0
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _run_bench_passes(
    compiled_module: Any,
    sample_inputs: list,
    n_passes: int,
    task: str,
) -> dict:
    """Run warm-up + timed inference passes on a forge-compiled module.

    Runs 2 warm-up passes (discarded) then n_passes timed passes.  Computes
    p50 and p95 inference latency and derives throughput from each percentile.
    Stops silently on error — returns partial results when some passes succeed,
    and an empty dict when all passes fail or n_passes is 0 or fewer.

    Args:
        compiled_module: forge-compiled module from forge.compile().
        sample_inputs:   Normalised input tensor list (as returned by _compile_model).
        n_passes:        Number of timed passes.  0 or negative returns {}.
        task:            HuggingFace task string — controls throughput unit via
                         _compute_throughput (tokens/sec for token tasks, ms/sample
                         for vision/audio/embeddings).

    Returns:
        A dict with keys:
          bench_passes  — actual number of timed passes completed (may be < n_passes
                          if the module crashed mid-way).
          infer_p50_s   — p50 (median) inference latency in seconds (rounded to 4 dp).
          infer_p95_s   — p95 inference latency in seconds (rounded to 4 dp).
          throughput_p50 — throughput derived from p50 latency (rounded to 2 dp).
          throughput_p95 — throughput derived from p95 latency (rounded to 2 dp).
        Returns {} on failure (n_passes <= 0, warm-up crash, or zero timed passes).
    """
    if n_passes <= 0:
        return {}

    last_output = None

    # Warm-up: 2 passes, not timed.  If the module crashes during warm-up we
    # return {} immediately — there's nothing useful to measure.
    for _ in range(2):
        try:
            compiled_module(*sample_inputs)
        except Exception:
            return {}

    # Timed passes — stop at first exception so partial results are still useful.
    times: list[float] = []
    for _ in range(n_passes):
        try:
            t0 = time.time()
            last_output = compiled_module(*sample_inputs)
            times.append(time.time() - t0)
        except Exception:
            break

    if not times:
        return {}

    times.sort()
    p50 = _percentile(times, 50)
    p95 = _percentile(times, 95)
    throughput_p50, _ = _compute_throughput(task, last_output, p50)
    throughput_p95, _ = _compute_throughput(task, last_output, p95)
    return {
        "bench_passes":    len(times),
        "infer_p50_s":     round(p50, 4),
        "infer_p95_s":     round(p95, 4),
        "throughput_p50":  round(throughput_p50, 2),
        "throughput_p95":  round(throughput_p95, 2),
    }


def _run_shape_sweep(
    compiled_module: Any,
    loader: Any,
    task: str,
    n_passes: int,
) -> list[dict]:
    """Run bench passes at alternative input shapes to characterise scaling.

    For token tasks: sweeps sequence lengths [128, 512] at batch=1.
    For vision or other tasks: sweeps image size 384×384 (typical when default
    was 224).  Shape-level failures are silently skipped so one bad shape never
    aborts the whole sweep.

    Args:
        compiled_module: forge-compiled module from forge.compile().
        loader:          Original model loader — currently unused, reserved for
                         loaders that expose a custom make_inputs(spec) method.
        task:            HuggingFace pipeline task string — controls sweep shapes
                         and throughput units.
        n_passes:        Number of timed passes per shape (no warm-up).

    Returns:
        A list of dicts.  Each dict contains the shape spec key(s) plus:
          infer_s    — median (p50) inference latency in seconds (rounded to 4 dp).
          throughput — throughput derived from p50 latency (rounded to 2 dp).
        Shapes that fail to run return no entry in the list.
    """
    import torch

    if task in _TOKEN_TASKS:
        sweep = [{"seq": 128}, {"seq": 512}]

        def make_inputs(spec: dict) -> list:
            return [torch.randint(0, 1000, (1, spec["seq"]))]
    else:
        sweep = [{"img_size": 384}]

        def make_inputs(spec: dict) -> list:
            return [torch.randn(1, 3, spec["img_size"], spec["img_size"])]

    results: list[dict] = []
    for spec in sweep:
        try:
            inputs = _normalise_inputs(make_inputs(spec))
            times: list[float] = []
            last_out = None

            # Warm-up: 2 passes, discarded.
            for _ in range(2):
                compiled_module(*inputs)

            # Timed passes.
            for _ in range(n_passes):
                t0 = time.time()
                last_out = compiled_module(*inputs)
                times.append(time.time() - t0)

            if times:
                times.sort()
                p50 = _percentile(times, 50)
                tput, _ = _compute_throughput(task, last_out, p50)
                results.append({
                    **spec,
                    "infer_s":    round(p50, 4),
                    "throughput": round(tput, 2),
                })
        except Exception:
            pass  # skip shapes that fail — don't abort the whole sweep

    return results


class TimeoutException(Exception):
    """Raised by the SIGALRM handler when a compile/inference step hangs."""
    pass


def _timeout_handler(signum, frame):
    """POSIX signal handler: converts SIGALRM into a TimeoutException."""
    raise TimeoutException("Operation timed out")


def _set_pane_title(title: str) -> None:
    """Push a live title into the tmux pane border via OSC 2 escape sequence.

    tmux intercepts OSC 2 and stores it as pane_title, which the border
    format #{pane_title} then displays.  Works silently in non-tmux terminals.
    """
    sys.stdout.write(f"\033]2;{title}\033\\")
    sys.stdout.flush()


def _decouple_stderr():
    """Silence fd2 for C++ noise — same technique as lib/worker.py.

    Uses FilteredStderr from lib/worker so the project doesn't duplicate
    the filtering logic. After this call, sys.stderr points to a
    FilteredStderr wrapping a dup of the original fd 2, and fd 2 itself
    is redirected to /dev/null so C++ libraries can't write noise.
    """
    import sys, os
    _project_root = str(Path(__file__).parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from lib.worker import FilteredStderr
    # Guard against double-patching across repeated calls.
    if isinstance(sys.stderr, FilteredStderr):
        return
    terminal_fd = os.dup(2)
    terminal_writer = os.fdopen(terminal_fd, "w", buffering=1, errors="replace")
    sys.stderr = FilteredStderr(terminal_writer)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)


def _print_rarity_reveal(model_id: str, rarity: str, newness: str,
                          task: str, source: str, is_first_ever: bool) -> None:
    """Print the big rarity-reveal banner for a model.

    Uses pyfiglet to render the short model name in ASCII art. Falls back
    gracefully to a plain bold line if pyfiglet is unavailable or the font
    fails. Pauses briefly after special (rare/legendary/zero-day) reveals.

    Args:
        model_id:      Full HuggingFace model identifier.
        rarity:        Rarity tier string (e.g. "legendary").
        newness:       Newness tier string (e.g. "zero_day").
        task:          HuggingFace pipeline task.
        source:        Data origin label.
        is_first_ever: True if this model has never compiled before.
    """
    rarity_color, rarity_label, rarity_pause = _RARITY_STYLE.get(
        rarity, (CYAN, "", 0.0))
    newness_color, newness_label, newness_pause = _NEWNESS_STYLE.get(
        newness, ("", "", 0.0))

    # The total dramatic pause is the maximum of rarity and newness pauses.
    pause = max(rarity_pause, newness_pause)

    # Collect badges to show above the ASCII banner.
    badges = []
    if is_first_ever:
        badges.append(f"{GOLD}⚡ FIRST EVER{RESET}")
    if newness_label:
        badges.append(f"{newness_color}{newness_label}{RESET}")
    if rarity_label:
        badges.append(f"{rarity_color}{rarity_label}{RESET}")

    print(f"\n{'─'*80}")
    if badges:
        print("  " + "  ".join(badges))

    # Seed model IDs use "name/task/backend" (e.g. "bge_1_5/embedding_generation/pytorch").
    # split("/")[-1] would give "pytorch" for those.  Strip known backend suffixes and
    # use the first path component, which is always the actual model name.
    _BACKENDS = {"pytorch", "jax", "onnx", "tensorflow", "flax", "paddle", "paddlepaddle"}
    parts = model_id.split("/")
    if parts[-1].lower() in _BACKENDS:
        short_name = parts[0]
    else:
        short_name = parts[-1]
    # Switch to "small" font for very long names to avoid terminal wrapping.
    font = "small" if len(short_name) > 25 else "standard"
    try:
        # Import inside the try so an ImportError (pyfiglet not installed) is
        # caught by the same except and falls through to the plain fallback.
        import pyfiglet
        banner = pyfiglet.figlet_format(short_name, font=font)
        print(f"{rarity_color or CYAN}{banner}{RESET}", end="")
    except Exception:
        # pyfiglet missing or font error — degrade gracefully.
        print(f"\n{BOLD}{rarity_color or CYAN}  {short_name}{RESET}\n")

    meta_parts = [task, source]
    print(f"  {DIM}{' · '.join(p for p in meta_parts if p)}{RESET}")

    if pause > 0:
        time.sleep(pause)


def _print_progress_step(step: int, total: int, desc: str, color=YELLOW) -> None:
    """Print a numbered progress step (e.g. [1/3] Loading model...)."""
    print(f"  {color}[{step}/{total}]{RESET} {desc}")


def _print_live_info(msg: str, ok: bool = True) -> None:
    """Print an indented info line with a ✓ (success) or → (neutral) marker."""
    marker = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}→{RESET}"
    print(f"    {marker} {msg}")


def _print_success(
    model_id: str,
    compile_time: float,
    infer_time: float,
    artifact: str,
    score_pts: int,
    is_first_ever: bool,
    streak: int,
    throughput: float = 0.0,
    throughput_unit: str = "",
    bench_p50: float = 0.0,
) -> None:
    """Print the success summary line after a successful compile + inference.

    Args:
        model_id:       HuggingFace model identifier.
        compile_time:   Seconds spent in forge.compile().
        infer_time:     Seconds spent on the first inference pass.
        artifact:       Decoded inference output string.
        score_pts:      Points awarded for this compilation event.
        is_first_ever:  True if this is the model's first-ever compilation.
        streak:         Current consecutive-success streak count.
        throughput:     Throughput value (tokens/sec or ms/sample). 0 = not shown.
        throughput_unit: "tokens/sec" or "ms/sample".
        bench_p50:      p50 infer time from bench passes (0 = bench not run).
    """
    streak_str = f"  {CYAN}🔥×{streak}{RESET}" if streak >= 3 else ""
    first_str  = f"  {GOLD}★ FIRST{RESET}" if is_first_ever else ""
    if bench_p50 > 0.0 and throughput_unit:
        tput_str = f"  ~{throughput:.1f} {throughput_unit} (p50)"
    elif throughput > 0.0 and throughput_unit:
        tput_str = f"  {throughput:.1f} {throughput_unit}"
    else:
        tput_str = ""
    print(f"\n  {BOLD}{GREEN}✓ SUCCESS{RESET}")
    print(f"    compile: {compile_time:.1f}s  infer: {infer_time:.2f}s"
          f"{tput_str}  pts: {GOLD}{score_pts:+d}{RESET}"
          f"{streak_str}{first_str}")
    print(f"    {DIM}{artifact[:80]}{RESET}")


def _print_failure(model_id: str, error: str, elapsed: float) -> None:
    """Print the failure summary line with a truncated error message and time.

    Args:
        model_id: Full model identifier (unused; kept for symmetry).
        error:    Error string (truncated to 80 chars to stay on one line).
        elapsed:  Total elapsed seconds before failure.
    """
    print(f"\n  {BOLD}{RED}✗ FAILED{RESET}  {DIM}{error[:80]}{RESET}  ({elapsed:.1f}s  −10pts)")


def _print_gated_pitch(
    model_id: str,
    downloads: int,
    pipeline_tag: str,
    gated_type: str,
    heading: str,
) -> None:
    """Print a gated-model pitch block with numbered unlock instructions.

    Called when the HF API confirms a model requires an access grant before
    its weights can be downloaded.  The pitch leads with what the model does
    and how many people use it (social proof), then hypes the Tenstorrent
    first-run angle, then gives a concrete numbered checklist to unlock it.

    Args:
        model_id:     HuggingFace model identifier (used to build the URL).
        downloads:    Total download count from HF for social proof.
        pipeline_tag: HF pipeline task tag (e.g. "text-generation").
        gated_type:   "manual" (fill-in form, may need approval wait) or
                      "auto" (one-click agree-and-access).
        heading:      extra_gated_heading from HF model card — the
                      human-readable reason for gating (may be "").
    """
    _TASK_LABELS = {
        "text-generation":              "a text-generation LLM",
        "text2text-generation":         "a seq2seq language model",
        "image-classification":         "an image classification model",
        "image-to-text":                "a vision-language model",
        "question-answering":           "a question-answering model",
        "fill-mask":                    "a masked language model",
        "token-classification":         "a token classification model",
        "feature-extraction":           "an embedding/feature model",
        "automatic-speech-recognition": "a speech recognition model",
        "audio-classification":         "an audio classification model",
    }
    task_label = _TASK_LABELS.get(pipeline_tag, f"a {pipeline_tag} model")
    dl_str = f"{downloads:,}" if downloads else "many"

    print(f"\n  {BOLD}{YELLOW}🔒  GATED MODEL{RESET}")
    print(f"  {'─'*56}")
    print(f"  {CYAN}What it is:{RESET}  {task_label}")
    print(f"  {CYAN}Downloads:{RESET}   {dl_str} devs already using this")
    if heading:
        print(f"  {CYAN}Access req:{RESET}  {heading[:72]}")
    print()
    print(f"  {GOLD}You could be the first to run this on Tenstorrent silicon.{RESET}")
    if gated_type == "manual":
        print(f"  {DIM}Approval required — may take minutes to a few hours:{RESET}")
    else:
        print(f"  {DIM}Takes ~2 minutes to unlock — here's how:{RESET}")
    print()
    print(f"  {BOLD}How to unlock:{RESET}")
    print(f"  {YELLOW}①{RESET}  Visit   {CYAN}https://huggingface.co/{model_id}{RESET}")
    if gated_type == "manual":
        print(f"  {YELLOW}②{RESET}  Fill in the access request form and submit")
        print(f"  {YELLOW}③{RESET}  Wait for approval (usually minutes to a few hours)")
        print(f"  {YELLOW}④{RESET}  Run    {CYAN}huggingface-cli login{RESET}")
        print(f"  {YELLOW}⑤{RESET}  Rerun  {CYAN}expedition with --allow-gated{RESET}")
    else:
        print(f"  {YELLOW}②{RESET}  Click  {CYAN}\"Agree and access repository\"{RESET}")
        print(f"  {YELLOW}③{RESET}  Run    {CYAN}huggingface-cli login{RESET}")
        print(f"  {YELLOW}④{RESET}  Rerun  {CYAN}expedition with --allow-gated{RESET}")
    print(f"  {'─'*56}")


def _preflight_gated_check(item: "QueueItem") -> tuple[bool, str]:
    """Check whether a HuggingFace model requires an access grant.

    Uses HfApi.model_info() to inspect the ``gated`` field *before* any
    download is attempted.  On a gated hit: prints the pitch block with
    unlock instructions (when stdin is a tty, also pauses 6 s to give the
    user time to read), then returns True so the caller can record a
    model_access failure and skip to the next model.

    Fails open on any API error (network blip, rate limit) — returns
    (False, "") so a transient HF API problem never blocks valid models.

    Runs for both frontier and seed models — seed models like command/pytorch
    and llama_3_2_vision require HF access grants and would otherwise waste a
    full download attempt before hitting a 403.

    Args:
        item: QueueItem for the model being preflighted.

    Returns:
        (is_gated, error_str) where error_str is non-empty only when gated.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(item.model_id)
        if not getattr(info, "gated", None):
            return False, ""

        gated_type = info.gated  # "manual" or "auto"
        downloads  = getattr(info, "downloads", 0) or 0
        card       = getattr(info, "card_data", None)
        heading    = ""
        if card is not None:
            heading = getattr(card, "extra_gated_heading", "") or ""

        _print_gated_pitch(
            model_id=item.model_id,
            downloads=downloads,
            pipeline_tag=item.task or "",
            gated_type=gated_type,
            heading=heading,
        )

        # Pause in interactive terminals so the user has time to read the pitch
        # before the next model's reveal fires.  Don't block indefinitely —
        # other chip panes are still running.
        if sys.stdin.isatty():
            time.sleep(6)

        return True, f"GatedModelError: model requires HF access grant (type={gated_type})"

    except Exception:
        return False, ""  # fail-open: API blip must not block valid models


# Custom module roots that are absent from the forge/XLA env.
# Key = importable root extracted from auto_map class path; Value = pip package name.
_CUSTOM_DEP_MAP: dict[str, str] = {
    "mamba_ssm":     "mamba-ssm",
    "simamba":       "simamba",
    "mamba":         "mamba-ssm",
    "FlagEmbedding": "FlagEmbedding",
    "rwkv":          "rwkv",
    "retnet":        "torchscale",
    "megalodon":     "megalodon",
}


def _preflight_arch_check(item: "QueueItem") -> tuple[bool, str]:
    """Fetch config.json only and detect unsupported architectures before any weight download.

    Catches models that use trust_remote_code with custom class dependencies that
    are not installed in the forge/XLA environment (mamba_ssm, simamba, RWKV, etc.).
    Config files are a few KB — this check costs ~1 second and zero weight bandwidth.

    Fails open on any exception (network blip, config absent) — a transient error
    must never block a valid model.  Only runs for frontier models.

    Returns:
        (False, "")        — architecture looks fine, or check inconclusive
        (True, error_str)  — architecture will fail; record and skip the model
    """
    if not item.is_frontier:
        return False, ""
    try:
        from huggingface_hub import hf_hub_download
        import json as _json

        config_path = hf_hub_download(item.model_id, "config.json",
                                      local_files_only=False)
        with open(config_path) as f:
            cfg = _json.load(f)

        auto_map = cfg.get("auto_map", {})
        for _key, class_path in auto_map.items():
            if not isinstance(class_path, str) or "." not in class_path:
                continue
            module_root = class_path.split(".")[0]
            if module_root in _CUSTOM_DEP_MAP:
                dep = _CUSTOM_DEP_MAP[module_root]
                return (True,
                        f"MissingDependency: requires '{dep}' "
                        f"(custom class {class_path!r}) which is not installed")
            try:
                __import__(module_root)
            except ImportError:
                return (True,
                        f"MissingDependency: custom class {class_path!r} "
                        f"requires importable module '{module_root}'")

        return False, ""
    except Exception:
        return False, ""   # fail-open


def _normalise_inputs(inputs: list) -> list:
    """Ensure tensors are contiguous float32 before forge sees them.

    Two problems forge is sensitive to:
    1. Non-contiguous tensors: PIL→numpy→.permute(2,0,1) produces strides like
       (3, 1, 672, 3) instead of the expected NCHW layout.  .contiguous() fixes this.
    2. Half-precision: models loaded with default_dtype=float16 or bfloat16 can fail
       forge's tracer with dtype-mismatch errors.  Casting to float32 before compile
       avoids this without changing the model itself.
    """
    import torch
    out = []
    for t in inputs:
        if isinstance(t, torch.Tensor):
            if t.dtype in (torch.float16, torch.bfloat16):
                t = t.to(torch.float32)
            t = t.contiguous()
        out.append(t)
    return out


def _try_install_missing(error_str: str) -> str | None:
    """If error_str is a ModuleNotFoundError, try to pip-install the package.

    Returns the package name that was installed (or attempted), or None if
    the error is not a missing-module error.
    """
    import re, subprocess
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_str)
    if not m:
        return None
    pkg = m.group(1).split(".")[0]  # top-level package name
    print(f"  {YELLOW}→ missing package '{pkg}' — trying pip install...{RESET}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            timeout=60, check=False,
        )
        print(f"  {GREEN}✓ installed '{pkg}' — retrying model{RESET}")
    except Exception as exc:
        print(f"  {RED}✗ pip install failed: {exc}{RESET}")
    return pkg


def _compile_model(model_loader, chip_id: int, timeout: int = 120) -> tuple[bool, Any, float, float, str, Any, list]:
    """Run forge compile + inference for one model.

    Returns a 7-tuple (success, output, compile_time, infer_time, error_str, compiled, sample_inputs):
      - success:       True if both compile and inference completed without error.
      - output:        The raw inference output tensor/list (None on failure).
      - compile_time:  Seconds spent in forge.compile() (0.0 on failure before compile).
      - infer_time:    Seconds spent in the inference call (0.0 on failure or timeout).
      - error_str:     Empty string on success; "TIMEOUT" or "ExcType: msg" on failure.
      - compiled:      The forge-compiled module (None on failure); callers may run
                       additional inference passes (First Voice) without recompiling.
      - sample_inputs: List of input tensors used for compile/inference ([] on failure).

    Compile is attempted twice for models that fail with a tracer type-inference
    error.  HuggingFace causal-LM models commonly return a CausalLMOutputWithPast
    or plain tuple (logits, past_kv) rather than a bare tensor; the forge tracer
    can't infer types for mixed-type tuples.  On first failure we retry with a
    _LogitsWrapper that strips the return to the primary tensor, giving the tracer
    a clean interface.  This converts a large class of frontier models from
    guaranteed failure to likely success at the cost of one extra compile attempt.

    Imports (torch, forge) are deferred so this module can be imported and the
    argument parser exercised without forge or torch installed — important for
    CI and the import-only verification step.

    Args:
        model_loader: Callable that returns a torch.nn.Module (already .eval()-able).
        chip_id:      Zero-based index of the TT chip to run on (passed to forge).
        timeout:      Maximum seconds to allow for the inference step (SIGALRM).
    """
    import torch

    sys.path.insert(0, os.path.expanduser("~/tt-forge-fe"))
    import forge

    # Wraps a model whose forward() returns a tuple/ModelOutput so the forge
    # tracer sees a single tensor.  Defined here so torch.nn.Module is in scope.
    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, *args, **kwargs):
            out = self.model(*args, **kwargs)
            if isinstance(out, (tuple, list)):
                return out[0]
            if hasattr(out, "logits"):
                return out.logits
            return out

    try:
        model = model_loader()

        # onnx.ModelProto objects don't have .eval() — only call it for
        # torch.nn.Module instances.
        try:
            import onnx as _onnx_mod
            _is_onnx = isinstance(model, _onnx_mod.ModelProto)
        except ImportError:
            _is_onnx = False
        if not _is_onnx:
            model.eval()

        # Patch the model config for forge traceability:
        # - use_cache=False: KV caches have dynamic shapes that the XLA runtime
        #   can't handle (causes INTERNAL error code 13); disabling them makes
        #   causal-LM outputs a single logits tensor instead of (logits, past_kv).
        # - return_dict=False: forge's TorchScript tracer requires tuple outputs,
        #   not HuggingFace ModelOutput dataclass instances.
        if hasattr(model, 'config') and not _is_onnx:
            cfg = model.config
            if hasattr(cfg, 'use_cache'):
                cfg.use_cache = False
            if hasattr(cfg, 'return_dict'):
                cfg.return_dict = False
            if hasattr(cfg, 'output_attentions'):
                cfg.output_attentions = False
            if hasattr(cfg, 'output_hidden_states'):
                cfg.output_hidden_states = False

        # Determine the input shape from the loader's optional _input_type hint.
        if hasattr(model_loader, "_input_type"):
            itype = model_loader._input_type
        else:
            itype = "image"  # default: vision model input shape

        # Prefer real inputs from the loader's load_inputs() when available —
        # these match the model's actual forward() signature and avoid shape
        # mismatches for encoder-decoder models (e.g. MusicGen) that need
        # multiple structured tensors.  Falls back to dummy on any error.
        sample_inputs: list = []
        _load_inputs_fn = getattr(model_loader, "_load_inputs", None)
        if _load_inputs_fn is not None:
            try:
                raw = _load_inputs_fn()
                if isinstance(raw, dict):
                    sample_inputs = [v for v in raw.values() if isinstance(v, torch.Tensor)]
                elif isinstance(raw, (list, tuple)):
                    sample_inputs = [v for v in raw if isinstance(v, torch.Tensor)]
                elif isinstance(raw, torch.Tensor):
                    sample_inputs = [raw]
            except Exception:
                sample_inputs = []

        if not sample_inputs:
            is_enc_dec = (hasattr(model, 'config') and
                          getattr(model.config, 'is_encoder_decoder', False))
            if is_enc_dec:
                # Encoder-decoder models (e.g. MusicGen, T5, BART) need both
                # input_ids for the encoder and decoder_input_ids for the decoder.
                # Providing only input_ids leaves decoder_input_ids=None and
                # causes forge to crash: "ones_like() argument must be Tensor,
                # not NoneType".
                sample_inputs = [
                    torch.randint(0, 1000, (1, 32)),   # input_ids
                    torch.randint(0, 1000, (1, 32)),   # decoder_input_ids
                ]
            elif itype == "text":
                # Minimal tokenized sequence: batch=1, seq_len=32, vocab_size=1000
                sample_inputs = [torch.randint(0, 1000, (1, 32))]
            elif itype == "audio":
                # 1 second of mono audio at 16kHz
                sample_inputs = [torch.randn(1, 16000)]
            else:
                # Standard ImageNet-style input: batch=1, RGB, 224×224
                sample_inputs = [torch.randn(1, 3, 224, 224)]

        # PIL→numpy→permute preprocessing can produce non-contiguous tensors whose
        # strides confuse the forge tracer (stride mismatch error on compile or inference).
        # Also cast float16/bfloat16 to float32 — forge requires float32 inputs.
        sample_inputs = _normalise_inputs(sample_inputs)

        _print_live_info(f"Architecture: {type(model).__name__}")

        compile_start = time.time()
        _print_progress_step(2, 3, "Compiling for TT hardware...")

        # ONNX models are compiled directly; PyTorch models go through the
        # two-stage wrapper retry to handle tuple/ModelOutput return types.
        compiled = None
        compile_error = None
        if _is_onnx:
            # forge.compile() accepts onnx.ModelProto directly.
            # ONNX models have fixed input shapes — use a 1×3×224×224 dummy
            # that matches the export shape used by most vision ONNX exporters.
            try:
                compiled = forge.compile(model, sample_inputs=sample_inputs)
            except Exception as exc:
                compile_error = exc
        else:
            # Two-stage compile: raw model first, then wrapped on tracer failure.
            for attempt, target in enumerate([model, _LogitsWrapper(model)]):
                try:
                    compiled = forge.compile(target, sample_inputs=sample_inputs)
                    compile_error = None
                    break
                except Exception as exc:
                    compile_error = exc
                    if attempt == 0 and "Tracer cannot infer type" in str(exc):
                        _print_live_info(
                            "Output is tuple — retrying with logits wrapper…", ok=False
                        )
                        continue
                    break  # non-tracer error: no point retrying

        compile_time = time.time() - compile_start

        if compile_error is not None:
            e = compile_error
            return False, None, compile_time, 0.0, f"{type(e).__name__}: {str(e)[:300]}", None, []

        _print_progress_step(3, 3, f"Running inference on chip {chip_id}...")
        # Use SIGALRM so that a hung inference doesn't stall the entire worker.
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            infer_inputs = _normalise_inputs(sample_inputs)
            infer_start = time.time()
            output = compiled(*infer_inputs)
            infer_time = time.time() - infer_start
            signal.alarm(0)  # cancel the alarm on clean completion
        except TimeoutException:
            signal.alarm(0)
            return False, None, compile_time, 0.0, "TIMEOUT", None, []

        # Unwrap list outputs (forge sometimes returns [tensor]).
        if isinstance(output, list):
            output = output[0] if output else None

        return True, output, compile_time, infer_time, "", compiled, list(sample_inputs)

    except TimeoutException as e:
        signal.alarm(0)
        return False, None, 0.0, 0.0, "TIMEOUT", None, []
    except Exception as e:
        signal.alarm(0)
        return False, None, 0.0, 0.0, f"{type(e).__name__}: {str(e)[:300]}", None, []


def _cleanup_dev_shm() -> None:
    """Remove stale forge shared memory segments from /dev/shm.

    forge.compile() creates sm_segment.tt-*.*.0 files in /dev/shm that persist
    after process crashes (SIGSEGV).  Accumulation of stale segments corrupts
    subsequent forge.compile() calls in parallel workers.  Called after each
    subprocess compile completes (success or crash) so the next model starts
    with a clean /dev/shm slate.
    """
    import glob
    for path in glob.glob("/dev/shm/sm_segment.tt-*.*.0"):
        try:
            os.remove(path)
        except OSError:
            pass  # already gone or not owned by this process


def _isolated_compile_worker(
    item_dict: dict,
    chip_id: int,
    result_path: str,
) -> None:
    """Subprocess target: compile one model and write a JSON result file.

    Runs the full pipeline (build loader → compile → decode → first voice →
    throughput) inside a child process so that a SIGSEGV inside forge.compile()
    kills only this child — the parent expedition worker survives and continues
    to the next model.

    All return values are serialised to JSON (result_path) so the parent can
    read them without pickling torch tensors or forge compiled modules.

    Args:
        item_dict:   Serialised QueueItem fields (all JSON-serialisable scalars).
        chip_id:     Zero-based TT chip index passed through to _compile_model.
        result_path: Path where this function writes its JSON result dict.
    """
    result: dict = {
        "success": False,
        "compile_time": 0.0,
        "infer_time": 0.0,
        "error_str": "",
        "artifact": "",
        "first_voice_text": "",
        "first_voice_sample": None,
        "throughput": 0.0,
        "throughput_unit": "",
    }
    try:
        from lib.expedition.decoder import decode, FrontierModelInfo

        item = QueueItem(**item_dict)
        loader = _build_loader(item)

        success, output, compile_time, infer_time, error_str, compiled_module, sample_inputs = (
            _compile_model(loader, chip_id)
        )
        # Mirror the auto-install retry from the parent — pip installs run inside
        # the subprocess so the installed package is available for the retry call.
        if not success and "No module named" in error_str:
            if _try_install_missing(error_str):
                success, output, compile_time, infer_time, error_str, compiled_module, sample_inputs = (
                    _compile_model(loader, chip_id)
                )

        result["compile_time"] = compile_time
        result["infer_time"] = infer_time
        result["error_str"] = error_str
        result["success"] = success

        if success:
            model_info = FrontierModelInfo(
                name=item.model_id, task=item.task, source=item.source
            )
            result["artifact"] = decode(output, model_info)

            fv_text, fv_sample = _attempt_first_voice(
                compiled_model=compiled_module,
                task=item.task,
                model_id=item.model_id,
            )
            result["first_voice_text"] = fv_text
            result["first_voice_sample"] = fv_sample

            throughput, throughput_unit = _compute_throughput(
                item.task, output, infer_time
            )
            result["throughput"] = throughput
            result["throughput_unit"] = throughput_unit

    except Exception as exc:
        result["error_str"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    finally:
        # Always attempt to write results so the parent can read them, even on
        # failure.  If this write itself fails, the parent detects a missing/empty
        # file and falls back to the default (failed) result.
        try:
            with open(result_path, "w") as _f:
                json.dump(result, _f)
        except Exception:
            pass


def _compile_isolated(item: "QueueItem", chip_id: int) -> dict:
    """Compile one model in a child process, isolating forge.compile() crashes.

    forge.compile() occasionally triggers SIGSEGV in the underlying C++ runtime.
    Running each compile in a fresh child process means a crash kills only that
    child; the parent expedition worker and all other parallel chip workers
    survive unaffected.

    After the child exits (cleanly or via SIGSEGV), any stale /dev/shm
    sm_segment.tt-* files left by that compile are cleaned up so the next model
    starts with a fresh shared memory state.

    Bench passes are not supported in isolated mode because the forge compiled
    module cannot be pickled across process boundaries.  bench_passes > 0 has
    no effect when called through this function.

    Args:
        item:    QueueItem describing the model to compile.
        chip_id: Zero-based TT chip index.

    Returns:
        dict with keys: success, compile_time, infer_time, error_str, artifact,
        first_voice_text, first_voice_sample, throughput, throughput_unit.
    """
    item_dict = asdict(item)

    # Write subprocess results to a temp file; avoids IPC pipe sizing limits
    # and allows the subprocess to flush incrementally.
    import tempfile
    result_fd, result_path = tempfile.mkstemp(prefix="forge_result_", suffix=".json")
    os.close(result_fd)

    proc = multiprocessing.Process(
        target=_isolated_compile_worker,
        args=(item_dict, chip_id, result_path),
        daemon=False,
    )
    proc.start()
    proc.join()  # blocks until child exits (normally or via signal)

    # Clean up stale forge shared memory regardless of how the child exited.
    _cleanup_dev_shm()

    exitcode = proc.exitcode  # 0 = clean exit; negative = killed by signal (-11 = SIGSEGV)

    default_result: dict = {
        "success": False,
        "compile_time": 0.0,
        "infer_time": 0.0,
        "error_str": "",
        "artifact": "",
        "first_voice_text": "",
        "first_voice_sample": None,
        "throughput": 0.0,
        "throughput_unit": "",
    }

    try:
        with open(result_path) as _f:
            content = _f.read().strip()
        if content:
            default_result.update(json.loads(content))
    except Exception:
        pass  # subprocess crashed before writing — keep default (failed) result
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    # Annotate SIGSEGV crashes that produced no error string so the bestiary
    # classifier can bucket them as forge_internal.
    if (
        exitcode is not None
        and exitcode < 0
        and not default_result["success"]
        and not default_result["error_str"]
    ):
        sig = -exitcode
        default_result["error_str"] = (
            f"SIGSEGV: forge.compile() killed by signal {sig} (forge_internal)"
        )

    return default_result


def _dispatch_xla_item(
    item: "QueueItem",
    chip_id: int,
    run_number: int,
    bestiary_path: str,
) -> dict:
    """Dispatch a JAX/Flax model to expedition_worker_xla.py and return compile result.

    The XLA worker subprocess handles bestiary recording, HUD writes, and
    result CSV creation internally.  This function collects only the compile
    result dict so the forge worker can update its own score/HUD tracking.

    Returns a dict compatible with _compile_isolated(): keys success,
    compile_time, infer_time, error_str, artifact, first_voice_text,
    first_voice_sample, throughput, throughput_unit.
    """
    import tempfile
    import subprocess as _sp

    project_dir = Path(__file__).resolve().parent.parent.parent
    xla_python = Path.home() / "tt-xla" / "venv" / "bin" / "python3"

    default: dict = {
        "success": False, "compile_time": 0.0, "infer_time": 0.0,
        "error_str": "", "artifact": "", "first_voice_text": "",
        "first_voice_sample": None, "throughput": 0.0, "throughput_unit": "",
    }

    if not xla_python.exists():
        default["error_str"] = "XLA venv not found at ~/tt-xla/venv — cannot compile JAX model"
        return default

    item_fd, item_path = tempfile.mkstemp(prefix="xla_item_", suffix=".json")
    res_fd,  res_path  = tempfile.mkstemp(prefix="xla_res_",  suffix=".csv")
    os.close(item_fd)
    os.close(res_fd)

    try:
        with open(item_path, "w") as f:
            json.dump(asdict(item), f, default=str)

        cmd = [
            str(xla_python),
            str(project_dir / "lib" / "expedition" / "expedition_worker_xla.py"),
            "--chip",       str(chip_id),
            "--run",        str(run_number),
            "--bestiary",   bestiary_path,
            "--model-json", item_path,
            "--results",    res_path,
        ]
        _sp.run(cmd, timeout=660)  # 11-min hard cap; XLA worker has its own 300s SIGALRM

        if Path(res_path).stat().st_size > 0:
            with open(res_path) as f:
                for row in csv.DictReader(f):
                    default["success"]          = row.get("status")            == "success"
                    default["compile_time"]     = float(row.get("compile_time") or 0)
                    default["infer_time"]       = float(row.get("infer_time")   or 0)
                    default["error_str"]        = row.get("error")              or ""
                    default["artifact"]         = row.get("artifact")           or ""
                    default["first_voice_text"] = row.get("first_voice")        or ""
                    break
    except _sp.TimeoutExpired:
        default["error_str"] = "XLA worker timeout (660s)"
    except Exception as exc:
        default["error_str"] = f"XLA dispatch error: {exc}"
    finally:
        for p in (item_path, res_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    return default


def _attempt_first_voice(
    compiled_model,
    task: str,
    model_id: str,
    timeout: int = 60,
) -> tuple[str, dict | None]:
    """Run a second 'First Voice' inference pass using a real themed sample input.

    Uses lib.expedition.sampler to pick a sample appropriate for the task,
    then runs the already-compiled model with it.  Attempts to decode the result
    into human-readable text.

    Returns a 2-tuple:
      (first_voice_text, sample_dict)
    where first_voice_text is the decoded output string (or "" on failure)
    and sample_dict is the sampler result used (or None on failure).

    This is a best-effort pass — all errors are suppressed so a failed First
    Voice never blocks the main compile result.
    """
    if compiled_model is None:
        return "", None

    try:
        import torch
        from lib.expedition.sampler import get_sample, make_tensor_input
        from lib.expedition.decoder import decode, FrontierModelInfo

        sample = get_sample(task)
        if sample is None:
            return "", None

        # Load the model's own tokenizer BEFORE make_tensor_input so text inputs
        # are encoded into the correct vocabulary.  Without this, make_tensor_input
        # returns (None, "no tokenizer") and the First Voice pass is skipped entirely.
        tokenizer = None
        if sample["input_type"] == "text":
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=True
                )
            except Exception:
                pass

        tensor_input, suffix = make_tensor_input(sample, seq_len=32, tokenizer=tokenizer)
        if tensor_input is None:
            return "", None

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            output = compiled_model(tensor_input)
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            return "", None

        if isinstance(output, list):
            output = output[0] if output else None
        if output is None:
            return "", None

        model_info = FrontierModelInfo(name=model_id, task=task, source="huggingface")
        # Pass the actual tensor_input as `inputs` so masked_lm and QA decoders
        # can locate the [MASK] position / decode the answer span from input_ids.
        text = decode(output, model_info, inputs=tensor_input, tokenizer=tokenizer)
        return text, sample

    except Exception:
        signal.alarm(0)
        return "", None


@dataclass
class QueueItem:
    """One entry in this chip's model queue, deserialized from the queue JSON.

    Matches the shape written by the expedition orchestrator when it partitions
    the full model list across chips.

    Attributes:
        model_id:     HuggingFace model identifier.
        display_name: Human-readable short name for display.
        task:         HuggingFace pipeline task string.
        source:       Data origin: "huggingface", "local", etc.
        rarity:       Pre-computed rarity tier string.
        hf_downloads: Total download count from HuggingFace (None if unavailable).
        hf_likes:     HuggingFace ♥ count (None if unavailable).
        hf_params_b:  Approximate parameter count in billions (None if unavailable).
        hf_created_at: ISO-8601 creation timestamp (None if unavailable).
        mesh_chips:   Number of TT chips required by this model.
        loader_module: Python module path for non-frontier models (may be None).
        loader_class:  Class name within loader_module (may be None).
        is_frontier:  True if this is a newly-discovered HuggingFace model (dynamic loader).
        library:      Model framework library ("pytorch", "jax", etc.; None = unknown).
        model_type:   HuggingFace model architecture type (None for seed models).
    """
    model_id: str
    display_name: str
    task: str
    source: str
    rarity: str
    hf_downloads: Optional[int]
    hf_created_at: Optional[str]
    mesh_chips: int
    loader_module: Optional[str]
    loader_class: Optional[str]
    is_frontier: bool = False
    hf_likes: Optional[int] = None
    hf_params_b: Optional[float] = None
    library: Optional[str] = None
    model_type: Optional[str] = None


def _load_queue(queue_path: str) -> list[QueueItem]:
    """Deserialize the chip queue JSON into a list of QueueItem dataclasses.

    Args:
        queue_path: Path to the JSON file produced by the expedition orchestrator.

    Returns:
        List of QueueItem instances in the order the orchestrator assigned them.
    """
    with open(queue_path) as f:
        items = json.load(f)
    return [QueueItem(**item) for item in items]


def _load_single_model(model_json_path: str) -> QueueItem:
    """Load a single model JSON written by the TUI dispatcher.

    The TUI can write a one-model JSON file per chip for per-model dispatch,
    allowing results to accumulate across multiple worker invocations in append
    mode rather than processing the whole queue in a single long-lived process.

    Args:
        model_json_path: Path to the single-model JSON file (a dict, not a list).

    Returns:
        A single QueueItem instance.
    """
    with open(model_json_path) as f:
        data = json.load(f)
    return QueueItem(**data)


def _build_loader(item: QueueItem):
    """Return a callable that loads the model described by item.

    Two code paths:
    - is_frontier=True: uses hf_discover.build_dynamic_loader for fresh HF models.
    - is_frontier=False: imports the model's dedicated loader class from
      tt-forge-models and wraps its load_model() in a closure.

    The returned callable has an optional _input_type attribute (str) that
    _compile_model uses to choose the correct sample input shape.

    Raises:
        ValueError: If a frontier model's dynamic loader cannot be constructed
                    (unsupported architecture, missing weights, etc.).
        ImportError: If the loader module or class is missing from tt-forge-models.

    Args:
        item: QueueItem describing the model to load.
    """
    if item.is_frontier:
        # Frontier models come from real-time HuggingFace discovery; use the
        # dynamic loader that auto-detects architecture from model config.
        from lib.expedition.hf_discover import FrontierModel, build_dynamic_loader
        from lib.expedition.scorer import Rarity, Newness, compute_rarity, compute_newness
        fm = FrontierModel(
            model_id=item.model_id,
            pipeline_tag=item.task,
            downloads=item.hf_downloads or 0,
            likes=item.hf_likes or 0,
            params_b=item.hf_params_b or 0.0,
            created_at=None,
            rarity=compute_rarity(item.hf_downloads),
            newness=Newness.ESTABLISHED,
            mesh_chips=item.mesh_chips,
        )
        loader = build_dynamic_loader(fm)
        if loader is None:
            raise ValueError(f"Cannot build dynamic loader for {item.model_id}")
        return loader
    else:
        # Known models have a dedicated loader class in the tt-forge-models repo.
        if item.loader_module is None or item.loader_class is None:
            raise ValueError(
                f"Non-frontier model {item.model_id!r} missing loader_module/loader_class"
            )

        # JAX/Flax models must not run through the forge (PyTorch) worker.
        # The router should have sent them to the XLA worker; if they arrive
        # here it's a misconfiguration — fail fast with a clear message.
        if (item.library or "").lower() in ("jax", "flax"):
            raise ValueError(
                f"JAX model {item.model_id!r} routed to forge worker — "
                "check backend setting (should be 'auto' or 'xla')"
            )

        import importlib, types
        forge_models_path = os.path.expanduser("~/code/tt-forge-models")
        # Loader modules are stored with a "_forgems." prefix so that their
        # relative imports resolve against the forge-models root (the directory
        # name has a hyphen and can't be a real Python package).
        _PKG = "_forgems"
        if _PKG not in sys.modules:
            root_mod = types.ModuleType(_PKG)
            root_mod.__path__ = [forge_models_path]
            root_mod.__package__ = _PKG
            root_mod.__file__ = os.path.join(forge_models_path, "__init__.py")
            sys.modules[_PKG] = root_mod
        mod = importlib.import_module(item.loader_module)
        cls = getattr(mod, item.loader_class)
        instance = cls()
        import inspect as _inspect, tempfile as _tempfile
        _sig = _inspect.signature(instance.load_model)
        if 'onnx_tmp_path' in _sig.parameters:
            _onnx_dir = _tempfile.mkdtemp(prefix='forge_onnx_')
            def loader(_p=_onnx_dir, _i=instance):
                return _i.load_model(onnx_tmp_path=_p)
        else:
            def loader(_i=instance):
                return _i.load_model()
        # Derive input type from task string first; fall back to the loader's
        # own _input_type hint, then to "image" for unlabelled vision models.
        # This prevents NLP models from receiving an image-shaped dummy tensor.
        task_lower = (item.task or "").lower()
        if any(x in task_lower for x in (
            "nlp", "text", "lm", "token", "qa", "squad", "masked",
            "causal", "seq2seq", "translation", "summarization",
            "sentiment", "ner", "classification",
        )) and "image" not in task_lower and "vision" not in task_lower:
            loader._input_type = "text"
        elif any(x in task_lower for x in ("audio", "speech", "wav", "asr")):
            loader._input_type = "audio"
        else:
            loader._input_type = getattr(instance, "_input_type", "image")
        # Attach load_inputs so _compile_model can get real structured inputs
        # for complex models (encoder-decoder, multi-input) instead of a dummy.
        if hasattr(instance, "load_inputs"):
            loader._load_inputs = instance.load_inputs
        return loader


def run_worker(chip_id: int, run_number: int, bestiary_path: str,
               queue_path: str | None, results_path: str,
               model_json_path: str | None = None,
               bench_passes: int = 0,
               bench_shapes: bool = False,
               ephemeral: bool = False,
               evict_failures: bool = False,
               run_total: int = 0) -> None:
    """Main entry point for the per-chip worker.

    Loads the queue, iterates over every model, runs the full pipeline
    (reveal → load → compile → decode → score → persist), and writes a CSV
    results file when done.

    This function is also importable from other modules for testing purposes;
    it does NOT call sys.exit, so callers can inspect return values or state.

    Args:
        chip_id:         Zero-based index of the TT chip this worker owns.
        run_number:      Sequential expedition run number (shown in UI and saved to bestiary).
        bestiary_path:   Path to the bestiary JSON file (created if absent).
        queue_path:      Path to the queue JSON file for this chip. Optional when
                         model_json_path is provided.
        results_path:    Path to write the per-chip CSV results file. Opens in append
                         mode so multiple per-model invocations accumulate results.
        model_json_path: Optional path to a single-model JSON file (a flat dict rather
                         than a list). When provided, overrides queue_path and processes
                         exactly one model. The TUI uses this for per-model dispatch.
        bench_passes:    Number of timed inference passes to run after each successful
                         compile (preceded by 2 warm-up passes). 0 disables bench.
        bench_shapes:    When True, also sweep alternative input shapes after bench
                         passes. Requires bench_passes > 0 to have any effect.
        ephemeral:       If True, evict net-new HF downloads after each model unless gold star.
        evict_failures:  If True and ephemeral, also evict weights for failed models.
    """
    import datetime
    from lib.expedition.bestiary import Bestiary
    from lib.expedition.hud import ChipHUD
    from lib.expedition.scorer import (
        compute_rarity, compute_newness, compute_score, Rarity, Newness,
    )

    _decouple_stderr()
    bestiary = Bestiary(path=bestiary_path)
    if model_json_path:
        # Per-model TUI dispatch: process a single model from a flat JSON dict.
        queue = [_load_single_model(model_json_path)]
    elif queue_path:
        # Normal batch dispatch: process the whole chip queue JSON (list of dicts).
        queue = _load_queue(queue_path)
    else:
        raise ValueError("Either queue_path or model_json_path must be provided")
    hud = ChipHUD(chip_id=chip_id, total_models=run_total or len(queue), run_number=run_number)
    # Write the status file immediately so the ScoreStrip doesn't read a stale
    # file while this worker is initializing.  When resuming mid-run (TUI
    # per-model dispatch), this writes the already-accumulated pts/successes so
    # the score strip stays correct between model invocations.
    hud.write_status()

    _set_pane_title(f"C{chip_id} · {len(queue)} queued · run #{run_number:03d}")
    print(f"\n{BOLD}{CYAN}{'═'*80}{RESET}")
    print(f"{BOLD}{CYAN}  EXPEDITION CHIP {chip_id}  ·  {len(queue)} models queued  ·  run #{run_number:03d}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*80}{RESET}\n")

    results: list[dict] = []
    # Snapshot cache before any downloads so we only evict what this run fetched.
    preexisting: frozenset[str] = frozenset()
    if ephemeral:
        from lib.expedition import cache_janitor as _janitor
        preexisting = _janitor.snapshot_preexisting()
    # Track the last decoded artifact so it can be teased at the top of the
    # next model's reveal block ("last: …") for narrative continuity.
    last_artifact = ""

    for idx, item in enumerate(queue, 1):
        # Update both the HUD status file and the pane border title so progress
        # is visible at a glance from the tmux layout without zooming in.
        hud.set_current(item.model_id, idx)
        hud.write_status()
        s = hud.state
        _BACKENDS = {"pytorch", "jax", "onnx", "tensorflow", "flax", "paddle", "paddlepaddle"}
        _parts = item.model_id.split("/")
        short_name = (_parts[0] if _parts[-1].lower() in _BACKENDS else _parts[-1])[:24]
        _set_pane_title(
            f"C{chip_id} [{idx}/{s.total_models}] {short_name}"
            f"  ✓{s.successes} ✗{s.failures}  {s.pts}pts"
        )

        is_first_ever = not bestiary.is_compiled(item.model_id)
        rarity = compute_rarity(item.hf_downloads)
        newness = compute_newness(item.hf_created_at, is_first_ever)

        _print_rarity_reveal(
            model_id=item.model_id,
            rarity=rarity.value,
            newness=newness.value,
            task=item.task,
            source=item.source,
            is_first_ever=is_first_ever,
        )

        # Show previous model's artifact as a teaser before the new compile starts.
        if last_artifact:
            print(f"  {DIM}last: {last_artifact}{RESET}")

        start = time.time()

        # ── Gated model preflight ────────────────────────────────────────────
        # Check HF access gate before attempting any download. Prints a pitch
        # + numbered unlock instructions on interactive terminals (6 s pause),
        # then records model_access and skips — never retry a gated model.
        # Runs for ALL models (seed + frontier) — seed models like command/pytorch
        # and llama_3_2_vision also require HF access grants.
        _is_gated, _gated_err = _preflight_gated_check(item)
        if _is_gated:
            elapsed = time.time() - start
            score = compute_score(False, is_first_ever, rarity, newness,
                                  hud.state.streak, mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            bestiary.record_failure(item.model_id, run_number, _gated_err)
            hud.write_status()
            bestiary.save()
            results.append({"model": item.model_id, "status": "failed",
                             "error": _gated_err, "pts": score.pts})
            continue

        # ── Architecture preflight ───────────────────────────────────────────
        # Fetch config.json only (a few KB) and check for custom-class deps
        # not installed in the forge/XLA env (mamba_ssm, simamba, RWKV, etc.).
        # Catches Simamba-style models before any multi-GB weight download.
        if item.is_frontier:
            _bad_arch, _arch_err = _preflight_arch_check(item)
            if _bad_arch:
                elapsed = time.time() - start
                score = compute_score(False, is_first_ever, rarity, newness,
                                      hud.state.streak, mesh_chips=item.mesh_chips)
                hud.record_failure(item.model_id)
                bestiary.record_failure(item.model_id, run_number, _arch_err)
                hud.write_status()
                bestiary.save()
                results.append({"model": item.model_id, "status": "failed",
                                 "error": _arch_err, "pts": score.pts})
                continue

        # ── JAX/Flax → XLA worker dispatch ──────────────────────────────────────
        # JAX models must compile via the XLA backend (expedition_worker_xla.py).
        # Detect by library tag or by the last path segment of the model ID (e.g.
        # "alexnet/image_classification/jax" ends in "jax").  Spawn the XLA worker
        # subprocess; it handles its own bestiary recording, HUD writes, and CSV.
        # The forge worker re-syncs bestiary from disk so subsequent models see the
        # XLA worker's updates, and tracks the result in its own HUD/score state.
        _item_is_jax = (item.library or "").lower() in ("jax", "flax") or \
                       item.model_id.split("/")[-1].lower() in ("jax", "flax")

        if _item_is_jax:
            _print_progress_step(2, 3, "Routing to XLA worker (JAX)...")
            cr = _dispatch_xla_item(item, chip_id, run_number, bestiary_path)
            # XLA worker updated bestiary on disk — reload so this worker's
            # in-memory state reflects those changes for subsequent models.
            bestiary = Bestiary(path=bestiary_path)
        else:
            _print_progress_step(1, 3, "Loading model...")

            # ── Loader construction (pre-flight check) ───────────────────────────
            # Build the loader in the parent as a fast error gate — catches missing
            # modules and bad configs before paying the cost of a subprocess spawn.
            # The subprocess rebuilds its own loader; this result is not passed across
            # the process boundary.
            try:
                _build_loader(item)
            except Exception as e:
                elapsed = time.time() - start
                _print_failure(item.model_id, str(e), elapsed)
                score = compute_score(False, is_first_ever, rarity, newness, hud.state.streak,
                                      mesh_chips=item.mesh_chips)
                hud.record_failure(item.model_id)
                bestiary.record_failure(item.model_id, run_number, str(e))
                hud.write_status()
                results.append({"model": item.model_id, "status": "failed",
                                "error": str(e), "pts": score.pts})
                continue

            # ── Compile + inference (subprocess-isolated) ─────────────────────────
            # Each compile runs in a child process so a SIGSEGV inside forge.compile()
            # kills only the child — this worker loop survives and continues to the
            # next model.  decode(), first_voice, and throughput all run inside the
            # subprocess; only JSON-serialisable scalars cross the boundary.
            # bench_passes are silently skipped (compiled module can't be pickled).
            cr = _compile_isolated(item, chip_id)
        success          = cr["success"]
        compile_time     = cr["compile_time"]
        infer_time       = cr["infer_time"]
        error_str        = cr["error_str"]
        artifact         = cr["artifact"]
        first_voice_text   = cr["first_voice_text"]
        first_voice_sample = cr["first_voice_sample"]
        throughput         = cr["throughput"]
        throughput_unit    = cr["throughput_unit"]
        elapsed = time.time() - start

        if success:
            last_artifact  = artifact
            is_first_voice = bool(first_voice_text)

            score = compute_score(success=True, is_first_ever=is_first_ever,
                                  rarity=rarity, newness=newness,
                                  streak=hud.state.streak,
                                  mesh_chips=item.mesh_chips,
                                  is_first_voice=is_first_voice)
            hud.record_success(item.model_id, score)

            _print_success(
                item.model_id, compile_time, infer_time, artifact,
                score.pts, is_first_ever, hud.state.streak,
                throughput=throughput, throughput_unit=throughput_unit,
            )

            # Print First Voice output if we got one.
            if is_first_voice and first_voice_sample:
                print(f"    {GOLD}🗣 First Voice{RESET}  "
                      f"{DIM}[{first_voice_sample['description']}]{RESET}")
                print(f"    {PINK}{first_voice_text}{RESET}")

            compiled_at = datetime.datetime.now().isoformat()

            # Write a field journal entry when First Voice succeeds.
            if is_first_voice and first_voice_sample:
                try:
                    from lib.expedition.notes import journal_entry
                    project_dir = Path(__file__).resolve().parent.parent.parent
                    journal_entry(
                        run_number=run_number,
                        chip_id=chip_id,
                        model_id=item.model_id,
                        task=item.task,
                        sample_description=first_voice_sample["description"],
                        first_voice_text=first_voice_text,
                        compile_time_s=compile_time,
                        score_pts=score.pts,
                        project_dir=project_dir,
                    )
                except Exception:
                    pass  # journal write failure must never interrupt the run

            # Persist the artifact text to data/artifacts/<safe_name>.txt.
            # When first voice succeeded, write the full decoded text; otherwise
            # write the raw tensor-stats artifact.
            bestiary.save_artifact(
                model_id=item.model_id,
                task=item.task,
                compiled_at=compiled_at,
                chip=chip_id,
                run=run_number,
                artifact_text=first_voice_text if is_first_voice else artifact,
            )

            # Record the compilation in the bestiary compiled dict.
            # XLA-dispatched items are already recorded by expedition_worker_xla.py;
            # skip to avoid double-writing (attempts counter, backend field, etc.).
            if not _item_is_jax:
                bestiary.record_success(
                    model_id=item.model_id,
                    chip=chip_id,
                    run=run_number,
                    time_s=compile_time + infer_time,
                    task=item.task,
                    source=item.source,
                    rarity=rarity.value,
                    hf_downloads=item.hf_downloads,
                    hf_created_at=item.hf_created_at,
                    artifact=artifact,
                    backend="forge",
                    first_voice=first_voice_text if is_first_voice else "",
                    compile_s=compile_time,
                    infer_s=infer_time,
                    throughput=throughput,
                    throughput_unit=throughput_unit,
                    mesh_chips=getattr(item, "mesh_chips", 1),
                )
                # Accumulate points into the per-chip all-time leaderboard entry.
                bestiary.add_chip_points(
                    chip=chip_id,
                    pts=score.pts,
                    first_ever=is_first_ever,
                    streak=hud.state.streak,
                )
                # Build and append the perf history record.
                perf_record: dict = {
                    "model_id":        item.model_id,
                    "run":             run_number,
                    "timestamp":       compiled_at,
                    "backend":         "forge",
                    "chip":            chip_id,
                    "compile_s":       round(compile_time, 4),
                    "infer_s":         round(infer_time, 4),
                    "throughput":      round(throughput, 3),
                    "throughput_unit": throughput_unit,
                }
                bestiary.append_perf_record(perf_record)
            results.append({
                "model":        item.model_id,
                "status":       "success",
                "pts":          score.pts,
                "compile_time": compile_time,
                "infer_time":   infer_time,
                "artifact":     first_voice_text if is_first_voice else artifact,
                "first_ever":   is_first_ever,
                "first_voice":  is_first_voice,
            })
        else:
            # Compile or inference failed — record the failure and deduct points.
            _print_failure(item.model_id, error_str, elapsed)
            score = compute_score(False, is_first_ever, rarity, newness,
                                  hud.state.streak, mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            # XLA-dispatched failures are already in the bestiary from the subprocess.
            if not _item_is_jax:
                bestiary.record_failure(item.model_id, run_number, error_str)
            results.append({"model": item.model_id, "status": "failed",
                            "error": error_str, "pts": score.pts})

        # Flush bestiary to disk after each model so a crash doesn't lose
        # all results. This is slightly slower but much safer for long runs.
        bestiary.save()
        # Ephemeral mode: evict net-new downloads unless gold star.
        if ephemeral:
            _evicted, _freed = _janitor.maybe_evict(
                item.model_id, score, preexisting, evict_failures
            )
            if _evicted:
                _mb = _freed / 1_048_576
                print(f"    {DIM}♻ {_mb:.0f} MB freed{RESET}")
            elif success and _janitor.is_gold_star(score):
                print(f"    {GOLD}★ SAVED{RESET}")
        hud.write_status()

    # ── Run complete ─────────────────────────────────────────────────────────
    hud.mark_done()
    hud.write_status()
    s = hud.state
    _set_pane_title(
        f"C{chip_id} DONE  ✓{s.successes} ✗{s.failures}  {s.pts}pts"
        f"  🔥×{s.best_streak}"
    )

    # Write the per-chip CSV results file so the orchestrator can aggregate.
    # Opened in append mode so that per-model TUI dispatch (multiple subprocess
    # calls for the same chip) accumulates rows rather than overwriting them.
    # The header guard checks whether the file is new/empty before writing the
    # header row so it appears exactly once even across repeated appends.
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_file_empty = not Path(results_path).exists() or Path(results_path).stat().st_size == 0
    with open(results_path, "a", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
            if results_file_empty:
                writer.writeheader()
            writer.writerows(results)

    # Print final summary — mirrors the HUD summary line for visibility in the
    # tmux pane that was running this worker.
    s = hud.state
    print(f"\n{BOLD}{GREEN}{'═'*80}{RESET}")
    print(f"{BOLD}CHIP {chip_id} DONE{RESET}  pts:{GOLD}{s.pts}{RESET}  "
          f"✓{s.successes} ✗{s.failures}  best streak: 🔥×{s.best_streak}")
    print(f"{BOLD}{GREEN}{'═'*80}{RESET}")
    # Keep the tmux pane open so the user can read the summary before the
    # window closes. The orchestrator's `wait` call will hold until Enter.
    try:
        input("Press Enter to close...")
    except EOFError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-chip Expedition worker: compile, decode, and score a model queue."
    )
    parser.add_argument("--chip",       type=int, required=True,
                        help="Zero-based index of the TT chip this worker owns.")
    parser.add_argument("--run",        type=int, required=True,
                        help="Sequential expedition run number.")
    parser.add_argument("--run-total",  type=int, default=0, metavar="N",
                        help="Total models assigned to this chip for the run. "
                             "Used by the progress bar; 0 = infer from queue length.")
    parser.add_argument("--bestiary",   default="data/bestiary.json",
                        help="Path to the bestiary JSON file.")
    parser.add_argument("--queue",      default=None,
                        help="Path to this chip's queue JSON file.")
    parser.add_argument("--model-json", default=None,
                        help="Path to a single-model JSON file. Overrides --queue.")
    parser.add_argument("--results",    required=True,
                        help="Path to write the per-chip CSV results file.")
    parser.add_argument("--bench-passes", type=int, default=0, metavar="N",
                        help="Run 2 warm-up + N timed inference passes after each "
                             "successful compile. Default 0 (disabled).")
    parser.add_argument("--bench-shapes", action="store_true",
                        help="Sweep alternative input shapes after bench passes. "
                             "Requires --bench-passes > 0.")
    parser.add_argument("--ephemeral", action="store_true",
                        help="Evict net-new HF model weights after each result "
                             "unless the model earns a gold-star rating.")
    parser.add_argument("--evict-failures", action="store_true",
                        help="With --ephemeral, also evict weights for failed models.")
    args = parser.parse_args()
    if not args.queue and not args.model_json:
        parser.error("one of --queue or --model-json is required")
    if args.queue and args.model_json:
        parser.error("--queue and --model-json are mutually exclusive")

    run_worker(
        chip_id=args.chip,
        run_number=args.run,
        bestiary_path=args.bestiary,
        queue_path=args.queue,
        model_json_path=args.model_json,
        results_path=args.results,
        bench_passes=args.bench_passes,
        bench_shapes=args.bench_shapes,
        ephemeral=args.ephemeral,
        evict_failures=args.evict_failures,
        run_total=args.run_total,
    )
