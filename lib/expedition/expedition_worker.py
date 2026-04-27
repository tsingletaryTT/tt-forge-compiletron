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
import argparse
import csv
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

# Silence C++ noise before any TT imports — same technique as lib/worker.py
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")

import warnings
warnings.filterwarnings("ignore")


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
_CSV_FIELDNAMES = ["model", "status", "pts", "compile_time", "artifact", "first_ever", "error"]


class TimeoutException(Exception):
    """Raised by the SIGALRM handler when a compile/inference step hangs."""
    pass


def _timeout_handler(signum, frame):
    """POSIX signal handler: converts SIGALRM into a TimeoutException."""
    raise TimeoutException("Operation timed out")


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

    # Use the portion after the last "/" as the display name for the banner.
    short_name = model_id.split("/")[-1]
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


def _print_success(model_id: str, compile_time: float, total_time: float,
                   artifact: str, score_pts: int, is_first_ever: bool,
                   streak: int) -> None:
    """Print the success summary line after a successful compile + inference.

    Shows compile time, total elapsed, points, first-ever badge, and streak
    emoji if the streak is 2 or longer. Also prints the first 120 chars of
    the decoded artifact (the model's "voice").

    Args:
        model_id:    Full model identifier (unused here; kept for symmetry with failure).
        compile_time: Seconds spent in forge.compile().
        total_time:   Total elapsed seconds including model loading.
        artifact:    Decoded inference output string.
        score_pts:   Points awarded for this compilation event.
        is_first_ever: True if this was the first-ever successful compile.
        streak:      Current consecutive success streak.
    """
    print(f"\n  {BOLD}{GREEN}✓ SUCCESS{RESET}")
    print(f"    compile: {compile_time:.1f}s  total: {total_time:.1f}s  "
          f"pts: {GOLD}{score_pts:+d}{RESET}"
          + (f"  {GOLD}★ FIRST EVER{RESET}" if is_first_ever else "")
          + (f"  🔥×{streak}" if streak >= 2 else ""))
    if artifact:
        print(f"    {CYAN}❝ {artifact[:120]}{RESET}")


def _print_failure(model_id: str, error: str, elapsed: float) -> None:
    """Print the failure summary line with a truncated error message and time.

    Args:
        model_id: Full model identifier (unused; kept for symmetry).
        error:    Error string (truncated to 80 chars to stay on one line).
        elapsed:  Total elapsed seconds before failure.
    """
    print(f"\n  {BOLD}{RED}✗ FAILED{RESET}  {DIM}{error[:80]}{RESET}  ({elapsed:.1f}s  −10pts)")


def _compile_model(model_loader, chip_id: int, timeout: int = 120) -> tuple[bool, Any, float, str]:
    """Run forge compile + inference for one model.

    Returns a 4-tuple (success, output, compile_time, error_str):
      - success:      True if both compile and inference completed without error.
      - output:       The raw inference output tensor/list (None on failure).
      - compile_time: Seconds spent in forge.compile() (0.0 on failure before compile).
      - error_str:    Empty string on success; "TIMEOUT" or "ExcType: msg" on failure.

    Imports (torch, forge) are deferred so this module can be imported and the
    argument parser exercised without forge or torch installed — important for
    CI and the Task 6 import-only verification step.

    Args:
        model_loader: Callable that returns a torch.nn.Module (already .eval()-able).
        chip_id:      Zero-based index of the TT chip to run on (passed to forge).
        timeout:      Maximum seconds to allow for the inference step (SIGALRM).
    """
    import torch

    sys.path.insert(0, os.path.expanduser("~/tt-forge-fe"))
    import forge

    try:
        model = model_loader()
        model.eval()

        # Determine the input shape from the loader's optional _input_type hint.
        if hasattr(model_loader, "_input_type"):
            itype = model_loader._input_type
        else:
            itype = "image"  # default: vision model input shape

        if itype == "text":
            # Minimal tokenized sequence: batch=1, seq_len=32, vocab_size=1000
            sample_input = torch.randint(0, 1000, (1, 32))
        elif itype == "audio":
            # 1 second of mono audio at 16kHz
            sample_input = torch.randn(1, 16000)
        else:
            # Standard ImageNet-style input: batch=1, RGB, 224×224
            sample_input = torch.randn(1, 3, 224, 224)

        _print_live_info(f"Architecture: {type(model).__name__}")

        compile_start = time.time()
        _print_progress_step(2, 3, "Compiling for TT hardware...")
        compiled = forge.compile(model, sample_inputs=[sample_input])
        compile_time = time.time() - compile_start

        _print_progress_step(3, 3, f"Running inference on chip {chip_id}...")
        # Use SIGALRM so that a hung inference doesn't stall the entire worker.
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            output = compiled(sample_input)
            signal.alarm(0)  # cancel the alarm on clean completion
        except TimeoutException:
            signal.alarm(0)
            return False, None, compile_time, "TIMEOUT"

        # Unwrap list outputs (forge sometimes returns [tensor]).
        if isinstance(output, list):
            output = output[0] if output else None

        return True, output, compile_time, ""

    except TimeoutException as e:
        signal.alarm(0)
        return False, None, 0.0, "TIMEOUT"
    except Exception as e:
        signal.alarm(0)
        return False, None, 0.0, f"{type(e).__name__}: {str(e)[:80]}"


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
        hf_downloads: Monthly download count from HuggingFace (None if unavailable).
        hf_created_at: ISO-8601 creation timestamp (None if unavailable).
        mesh_chips:   Number of TT chips required by this model.
        loader_module: Python module path for non-frontier models (may be None).
        loader_class:  Class name within loader_module (may be None).
        is_frontier:  True if this is a newly-discovered HuggingFace model (dynamic loader).
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
        import importlib
        forge_models_path = os.path.expanduser("~/code/tt-forge-models")
        if forge_models_path not in sys.path:
            sys.path.insert(0, forge_models_path)
        mod = importlib.import_module(item.loader_module)
        cls = getattr(mod, item.loader_class)
        instance = cls()
        def loader():
            return instance.load_model()
        # Prefer the loader instance's own _input_type if declared; fall back to
        # "image" since most tt-forge-models loaders are vision models.
        loader._input_type = getattr(instance, "_input_type", "image")
        return loader


def run_worker(chip_id: int, run_number: int, bestiary_path: str,
               queue_path: str, results_path: str) -> None:
    """Main entry point for the per-chip worker.

    Loads the queue, iterates over every model, runs the full pipeline
    (reveal → load → compile → decode → score → persist), and writes a CSV
    results file when done.

    This function is also importable from other modules for testing purposes;
    it does NOT call sys.exit, so callers can inspect return values or state.

    Args:
        chip_id:       Zero-based index of the TT chip this worker owns.
        run_number:    Sequential expedition run number (shown in UI and saved to bestiary).
        bestiary_path: Path to the bestiary JSON file (created if absent).
        queue_path:    Path to the queue JSON file for this chip.
        results_path:  Path to write the per-chip CSV results file.
    """
    import datetime
    from lib.expedition.bestiary import Bestiary
    from lib.expedition.decoder import decode, FrontierModelInfo
    from lib.expedition.hud import ChipHUD
    from lib.expedition.scorer import (
        compute_rarity, compute_newness, compute_score, Rarity, Newness,
    )

    _decouple_stderr()
    bestiary = Bestiary(path=bestiary_path)
    queue = _load_queue(queue_path)
    hud = ChipHUD(chip_id=chip_id, total_models=len(queue))

    print(f"\n{BOLD}{CYAN}{'═'*80}{RESET}")
    print(f"{BOLD}{CYAN}  EXPEDITION CHIP {chip_id}  ·  {len(queue)} models queued  ·  run #{run_number:03d}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*80}{RESET}\n")

    results: list[dict] = []
    # Track the last decoded artifact so it can be teased at the top of the
    # next model's reveal block ("last: …") for narrative continuity.
    last_artifact = ""

    for idx, item in enumerate(queue, 1):
        # Update the HUD status file immediately so the status pane shows the
        # currently-processing model even if compilation takes several minutes.
        hud.set_current(item.model_id, idx)
        hud.write_status()

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
            print(f"  {DIM}last: {last_artifact[:80]}{RESET}")

        _print_progress_step(1, 3, "Loading model...")
        start = time.time()

        # ── Loader construction ──────────────────────────────────────────────
        # This can fail for frontier models whose architecture we can't handle
        # or for tt-forge-models entries with missing dependencies.
        try:
            loader = _build_loader(item)
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

        # ── Compile + inference ──────────────────────────────────────────────
        success, output, compile_time, error_str = _compile_model(loader, chip_id)
        elapsed = time.time() - start

        if success:
            # Decode the raw tensor output into a human-readable artifact string.
            model_info = FrontierModelInfo(name=item.model_id, task=item.task,
                                           source=item.source)
            artifact = decode(output, model_info)
            last_artifact = artifact

            score = compute_score(success=True, is_first_ever=is_first_ever,
                                  rarity=rarity, newness=newness,
                                  streak=hud.state.streak,
                                  mesh_chips=item.mesh_chips)
            hud.record_success(item.model_id, score)

            _print_success(item.model_id, compile_time, elapsed, artifact,
                           score.pts, is_first_ever, hud.state.streak)

            compiled_at = datetime.datetime.now().isoformat()

            # Persist the artifact text to data/artifacts/<safe_name>.txt.
            # The default artifacts_dir in Bestiary.save_artifact is "data/artifacts".
            bestiary.save_artifact(
                model_id=item.model_id,
                task=item.task,
                compiled_at=compiled_at,
                chip=chip_id,
                run=run_number,
                artifact_text=artifact,
            )

            # Record the compilation in the bestiary compiled dict.
            bestiary.record_success(
                model_id=item.model_id,
                chip=chip_id,
                run=run_number,
                time_s=compile_time,
                task=item.task,
                source=item.source,
                rarity=rarity.value,
                hf_downloads=item.hf_downloads,
                hf_created_at=item.hf_created_at,
                artifact=artifact,
            )
            # Accumulate points into the per-chip all-time leaderboard entry.
            bestiary.add_chip_points(
                chip=chip_id,
                pts=score.pts,
                first_ever=is_first_ever,
                streak=hud.state.streak,
            )
            results.append({"model": item.model_id, "status": "success",
                            "pts": score.pts, "compile_time": compile_time,
                            "artifact": artifact, "first_ever": is_first_ever})
        else:
            # Compile or inference failed — record the failure and deduct points.
            _print_failure(item.model_id, error_str, elapsed)
            score = compute_score(False, is_first_ever, rarity, newness,
                                  hud.state.streak, mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            bestiary.record_failure(item.model_id, run_number, error_str)
            results.append({"model": item.model_id, "status": "failed",
                            "error": error_str, "pts": score.pts})

        # Flush bestiary to disk after each model so a crash doesn't lose
        # all results. This is slightly slower but much safer for long runs.
        bestiary.save()
        hud.write_status()

    # ── Run complete ─────────────────────────────────────────────────────────
    hud.mark_done()
    hud.write_status()

    # Write the per-chip CSV results file so the orchestrator can aggregate.
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
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
    input("Press Enter to close...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-chip Expedition worker: compile, decode, and score a model queue."
    )
    parser.add_argument("--chip",     type=int, required=True,
                        help="Zero-based index of the TT chip this worker owns.")
    parser.add_argument("--run",      type=int, required=True,
                        help="Sequential expedition run number.")
    parser.add_argument("--bestiary", default="data/bestiary.json",
                        help="Path to the bestiary JSON file.")
    parser.add_argument("--queue",    required=True,
                        help="Path to this chip's queue JSON file.")
    parser.add_argument("--results",  required=True,
                        help="Path to write the per-chip CSV results file.")
    args = parser.parse_args()

    run_worker(
        chip_id=args.chip,
        run_number=args.run,
        bestiary_path=args.bestiary,
        queue_path=args.queue,
        results_path=args.results,
    )
