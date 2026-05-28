#!/usr/bin/env python3
# lib/expedition/expedition_worker_xla.py
"""
Per-chip XLA expedition worker. Runs the JAX/PJRT compile pipeline for each
model in this chip's queue, then pipes results through decoder → scorer → hud.

Invoked by the expedition orchestrator (TUI or CLI) as:
  ~/tt-xla/venv/bin/python3 lib/expedition/expedition_worker_xla.py \
      --chip N --run R --bestiary data/bestiary.json \
      --queue /tmp/expedition_queue_chipN.json \
      --results /tmp/expedition_results_chipN.csv

Key differences from expedition_worker.py (forge):
  - Uses JAX + Flax via the pjrt-plugin-tt PJRT backend instead of forge.compile()
  - "Compilation" happens on the first jax.jit call (XLA JIT, not TorchScript)
  - No _LogitsWrapper needed — JAX handles model output structures differently
  - Seed models load via tt-forge-models *.jax.loader (Flax) not *.pytorch.loader
  - Frontier HuggingFace models use FlaxAutoModelForCausalLM etc.
  - Results CSV has an extra backend="xla" column
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_project_root = str(_Path(__file__).resolve().parent.parent.parent)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import argparse
import csv
import json
import os
import signal
import sys
import time
import collections.abc
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

# ── JAX 0.7.x / Flax 0.8.x compatibility patches ─────────────────────────────
# Two issues with pjrt-plugin-tt 0.9.0 + JAX 0.7.1 + Flax 0.8.5:
#
# 1. Flax's trace_level() calls main.level, but JAX 0.7.x removed that
#    attribute from Trace objects. Patch before any Flax/JAX model imports.
#
# 2. pjrt-plugin-tt only exposes "tt" backend. Transformers' from_pretrained
#    tries to put loaded weights on the "cpu" backend first; redirect to "tt".
#
# 3. Flax Module.init runs eagerly and hits SliceOp failures in eager mode.
#    Use _do_init=False in from_pretrained to skip eager init; JIT inference
#    compiles SliceOps via XLA which works correctly on TT.

def _apply_jax_compat_patches():
    try:
        import flax.core.tracers as _fct
        def _patched_trace_level(main):
            if main is None:
                return float('-inf')
            if hasattr(main, 'level'):
                return main.level
            # Derive nesting depth from parent_trace chain (JAX 0.7.x equivalent)
            level = 0
            t = main
            while (pt := getattr(t, 'parent_trace', None)) is not None:
                level += 1
                t = pt
            return level
        _fct.trace_level = _patched_trace_level
    except Exception:
        pass

    try:
        import jax._src.xla_bridge as _xb
        _orig_local_devices = _xb.local_devices
        def _patched_local_devices(process_index=None, backend=None):
            try:
                return _orig_local_devices(process_index=process_index, backend=backend)
            except RuntimeError:
                # CPU not available (JAX_PLATFORMS=tt); fall back to tt
                return _orig_local_devices(process_index=process_index, backend='tt')
        _xb.local_devices = _patched_local_devices
        import jax
        jax.local_devices = _patched_local_devices
    except Exception:
        pass

_apply_jax_compat_patches()

# ── ANSI colors ───────────────────────────────────────────────────────────────

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
TEAL   = "\033[38;5;43m"   # XLA accent — distinct from forge's color palette

_RARITY_STYLE = {
    "legendary": (PURPLE, "★★★ LEGENDARY", 2.0),
    "rare":      (PINK,   "★ RARE FIND",   1.0),
    "uncommon":  (YELLOW, "◆ UNCOMMON",     0.5),
    "common":    (CYAN,   "",               0.0),
    "familiar":  (CYAN,   "",               0.0),
}
_NEWNESS_STYLE = {
    "zero_day":    (GOLD,   "⚡ ZERO DAY",  3.0),
    "hot":         (YELLOW, "🔥 HOT",        0.5),
    "fresh":       (GREEN,  "✨ FRESH",       0.0),
    "established": ("",     "",              0.0),
    "familiar":    ("",     "",              0.0),
}

_CSV_FIELDNAMES = [
    "model", "status", "pts", "compile_time", "infer_time",
    "artifact", "first_ever", "first_voice", "error", "backend",
]

BACKEND_LABEL = "xla"

# Minimum sequence length for compilation and benchmarking.  Loaders that
# produce shorter inputs (e.g. BLOOM tokenising a 4-token sentence) are padded
# up to this length so timing numbers reflect a realistic workload.
_BENCH_SEQ_LEN = 32
# Sample text injected into loaders that expose a sample_text attribute.
# Long enough to hit _BENCH_SEQ_LEN on most tokenisers.
_BENCH_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning was the word, and the word was with God."
)

_TOKEN_TASKS: frozenset[str] = frozenset({
    "text-generation", "nlp_causal_lm", "nlp_masked_lm",
    "fill-mask", "nlp_text_cls", "nlp_token_cls",
})


def _compute_throughput_xla(task: str, output: Any, infer_s: float, batch: int = 1) -> tuple[float, str]:
    """Compute throughput for XLA output (JAX arrays).

    Same logic as the forge worker but handles JAX array shapes.
    batch > 1 for multi-chip runs where N examples are processed simultaneously.
    Returns (tokens_per_sec, "tokens/sec") for token tasks,
    (ms_per_sample, "ms/sample") otherwise.  Returns (0.0, "") on failure.
    """
    if infer_s <= 0.0 or output is None:
        return 0.0, ""
    try:
        if task in _TOKEN_TASKS:
            seq_len = output.shape[1]
            return round(batch * seq_len / infer_s, 3), "tokens/sec"
    except (AttributeError, IndexError):
        pass
    return round(infer_s * 1000.0 / max(batch, 1), 3), "ms/sample"


def _percentile_xla(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile of an already-sorted list of floats."""
    if not sorted_data:
        return 0.0
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _run_bench_passes_xla(
    compiled_bundle: tuple,
    task: str,
    n_passes: int,
) -> dict:
    """Run warm-up + timed inference passes using a JAX compiled bundle.

    compiled_bundle is (compiled_fn, params, device, (n_chips, dummy_inputs)).
    n_chips > 1 means multi-chip: skip jax.default_device and use the already-sharded
    dummy_inputs directly.  n_chips == 1 uses the single-device context path.
    Warm-up: 2 passes. Then n_passes timed.  Stops on error.
    Returns dict with bench_passes, infer_p50_s, infer_p95_s,
    throughput_p50, throughput_p95.  Empty dict on failure or n_passes <= 0.
    """
    if n_passes <= 0 or compiled_bundle is None:
        return {}
    try:
        compiled_fn, params, device, bench_info = compiled_bundle
        n_chips, dummy = bench_info
        import jax, jax.numpy as jnp

        def _infer(inputs):
            if n_chips > 1:
                # Multi-chip: compiled_fn has mesh-based dispatch; no device context.
                out = compiled_fn(params, inputs)
            else:
                with jax.default_device(device):
                    out = compiled_fn(params, inputs)
            out.block_until_ready()
            return out

        for _ in range(2):  # warm-up
            _infer(dummy)

        times: list[float] = []
        last_out = None
        for _ in range(n_passes):
            try:
                t0 = time.time()
                last_out = _infer(dummy)
                times.append(time.time() - t0)
            except Exception:
                break

        if not times:
            return {}
        times.sort()
        p50 = _percentile_xla(times, 50)
        p95 = _percentile_xla(times, 95)
        tput_p50, _ = _compute_throughput_xla(task, last_out, p50, batch=n_chips)
        tput_p95, _ = _compute_throughput_xla(task, last_out, p95, batch=n_chips)
        return {
            "bench_passes":   len(times),
            "infer_p50_s":    round(p50, 4),
            "infer_p95_s":    round(p95, 4),
            "throughput_p50": round(tput_p50, 2),
            "throughput_p95": round(tput_p95, 2),
        }
    except Exception:
        return {}


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def _set_pane_title(title: str) -> None:
    sys.stdout.write(f"\033]2;{title}\033\\")
    sys.stdout.flush()


def _print_rarity_reveal(model_id: str, rarity: str, newness: str,
                          task: str, source: str, is_first_ever: bool) -> None:
    rarity_color, rarity_label, rarity_pause = _RARITY_STYLE.get(rarity, (CYAN, "", 0.0))
    newness_color, newness_label, newness_pause = _NEWNESS_STYLE.get(newness, ("", "", 0.0))
    pause = max(rarity_pause, newness_pause)

    badges = []
    if is_first_ever:
        badges.append(f"{GOLD}⚡ FIRST EVER{RESET}")
    if newness_label:
        badges.append(f"{newness_color}{newness_label}{RESET}")
    if rarity_label:
        badges.append(f"{rarity_color}{rarity_label}{RESET}")
    badges.append(f"{TEAL}[XLA]{RESET}")

    print(f"\n{'─'*80}")
    if badges:
        print("  " + "  ".join(badges))

    _BACKENDS = {"pytorch", "jax", "onnx", "tensorflow", "flax", "paddle", "paddlepaddle"}
    parts = model_id.split("/")
    short_name = parts[0] if parts[-1].lower() in _BACKENDS else parts[-1]
    font = "small" if len(short_name) > 25 else "standard"
    try:
        import pyfiglet
        banner = pyfiglet.figlet_format(short_name, font=font)
        print(f"{rarity_color or TEAL}{banner}{RESET}", end="")
    except Exception:
        print(f"\n{BOLD}{rarity_color or TEAL}  {short_name}{RESET}\n")

    print(f"  {DIM}{task} · {source} · JAX/XLA{RESET}")
    if pause > 0:
        time.sleep(pause)


def _print_progress_step(step: int, total: int, desc: str, color=YELLOW) -> None:
    print(f"  {color}[{step}/{total}]{RESET} {desc}")


def _print_live_info(msg: str, ok: bool = True) -> None:
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
    """Print the post-compile success summary line for one XLA model.

    Shows compile time, infer time, optional throughput (from bench_p50 if
    available, otherwise from the single second-pass infer_time), score points,
    streak indicator, and first-ever badge.
    """
    streak_str = f"  {TEAL}🔥×{streak}{RESET}" if streak >= 3 else ""
    first_str  = f"  {GOLD}★ FIRST{RESET}" if is_first_ever else ""
    if bench_p50 > 0.0 and throughput_unit:
        tput_str = f"  ~{throughput:.1f} {throughput_unit} (p50)"
    elif throughput > 0.0 and throughput_unit:
        tput_str = f"  {throughput:.1f} {throughput_unit}"
    else:
        tput_str = ""
    print(f"\n  {BOLD}{GREEN}✓ SUCCESS{RESET}  {TEAL}[XLA]{RESET}")
    print(f"    compile: {compile_time:.1f}s  infer: {infer_time:.2f}s"
          f"{tput_str}  pts: {GOLD}{score_pts:+d}{RESET}"
          f"{streak_str}{first_str}")
    print(f"    {DIM}{artifact[:80]}{RESET}")


def _print_failure(model_id: str, error: str, elapsed: float) -> None:
    print(f"\n  {BOLD}{RED}✗ FAILED{RESET}  {DIM}{error[:80]}{RESET}  ({elapsed:.1f}s  −10pts)")


def _try_install_missing(error_str: str) -> str | None:
    """If error_str is a ModuleNotFoundError, pip-install the package and return its name."""
    import re, subprocess
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_str)
    if not m:
        return None
    pkg = m.group(1).split(".")[0]
    print(f"  {YELLOW}→ missing package '{pkg}' — trying pip install into XLA venv...{RESET}")
    # Install into the XLA venv (this process's interpreter)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                       timeout=60, check=False)
        print(f"  {GREEN}✓ installed '{pkg}' — retrying model{RESET}")
    except Exception as exc:
        print(f"  {RED}✗ pip install failed: {exc}{RESET}")
    return pkg


# ── JAX/XLA device setup ──────────────────────────────────────────────────────

def _setup_jax(chip_id: int):
    """Register the TT PJRT plugin and return the device for this chip.

    Must be called before any JAX operations. Unsets forge env vars that
    interfere with the XLA runtime (they use different TT-Metal builds).

    Returns the jax.Device for chip_id, or raises if no TT devices found.
    """
    # TT-XLA bundles its own TT-Metal — forge's TT_METAL_HOME would conflict.
    for var in ("TT_METAL_HOME", "TT_METAL_LOGGER_LEVEL"):
        os.environ.pop(var, None)

    # Silence XLA/JAX startup noise before importing.
    os.environ.setdefault("JAX_PLATFORMS", "tt")
    os.environ.setdefault("XLA_FLAGS", "--xla_dump_to=/dev/null")

    _print_live_info(f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES', '(not set)')}")
    import jax
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX/TT devices found — is pjrt-plugin-tt installed?")

    # Each chip worker owns a single device. If chip_id exceeds available
    # devices, fall back to device 0 (single-chip systems).
    device = devices[min(chip_id, len(devices) - 1)]
    _print_live_info(f"JAX device: {device}  ({len(devices)} total TT device(s))")
    return device


# ── XLA compile + inference ───────────────────────────────────────────────────

def _compile_model_xla(
    model_loader,
    device,
    chip_id: int,
    timeout: int = 300,
    mesh_chips: int = 1,
    meta: "Optional[_LoaderMeta]" = None,
) -> tuple[bool, Any, float, float, str, Any]:
    """Run JAX JIT compile + inference for one Flax model.

    Returns 6-tuple (success, output, compile_time, infer_s, error_str, compiled_bundle):
      - success:        True if compile and first inference succeeded.
      - output:         Raw JAX array output (None on failure).
      - compile_time:   Seconds for the first jax.jit call (JIT + first inference combined).
      - infer_s:        Seconds for a *second* inference pass run after JIT; measures
                        pure inference latency without compilation overhead.  0.0 if the
                        second pass was skipped or raised an exception.
      - error_str:      Empty on success; "TIMEOUT" or "ExcType: msg" on failure.
      - compiled_bundle: (compiled_fn, params, lead_device, (n_chips, dummy_inputs)) tuple
                        for bench passes and First Voice re-use.  None on failure.

    Unlike forge, there is no explicit "compile then run" step — jax.jit
    traces lazily and compiles on the first call. The compile_time reported
    here is the wall-clock duration of that first call, which includes both
    XLA compilation and the initial inference. infer_s measures a separate
    second call after JIT is warm, giving a cleaner inference-only figure.

    When mesh_chips > 1, uses JAX data parallelism: params are replicated
    across all chips and the input batch is sharded evenly across devices.
    This is genuine multi-chip computation — each device runs 1/N of the batch
    simultaneously and XLA handles cross-device communication automatically.
    Type A (Linen) models route through shard_map when meta.use_shard_map is True.

    Args:
        model_loader: Callable returning (flax_model, params, tokenizer, input_fn).
                      input_fn(device) → jax.Array dummy input.
        device:       The jax.Device this worker owns (lead chip for multi-chip).
        chip_id:      Zero-based chip index (for logging).
        timeout:      SIGALRM timeout in seconds for the compile+run step.
        mesh_chips:   Number of chips to use.  >1 triggers data-parallel sharding.
        meta:         Routing metadata from _build_loader_xla. Controls whether to
                      use shard_map (Type A Linen) or jit(in_shardings=...) (Type B).
    """
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        _print_live_info(f"Loading Flax model...")
        model, params, tokenizer, make_input = model_loader()

        _print_live_info(f"Architecture: {type(model).__name__}")

        # Linen models that don't use from_pretrained (e.g. AlexNet/custom) return
        # an empty params dict because .init() was never called. Use jax.eval_shape
        # to trace parameter shapes without device execution (TT doesn't support
        # eager mode, so model.init() would fail). Then zero-init the param tree —
        # good enough for the compile trace even though the weights aren't trained.
        if not params and hasattr(model, "init"):
            _print_live_info("params empty — attempting shape-based init via jax.eval_shape...")
            try:
                import jax
                import numpy as np
                _dummy_for_init = make_input(device)
                if isinstance(_dummy_for_init, _Mapping):
                    _shape_result = jax.eval_shape(
                        model.init, jax.random.PRNGKey(0), **_dummy_for_init
                    )
                else:
                    _shape_result = jax.eval_shape(
                        model.init, jax.random.PRNGKey(0), _dummy_for_init
                    )
                params = jax.tree_util.tree_map(
                    lambda s: np.zeros(s.shape, dtype=np.float32), _shape_result
                )
                _print_live_info(
                    f"shape-based init OK — "
                    f"keys: {list(params.keys()) if isinstance(params, dict) else 'flat'}"
                )
            except Exception as _ie:
                _print_live_info(
                    f"shape-based init failed: {str(_ie)[:120]}", ok=False
                )

        # Build the forward function to JIT. Four calling conventions:
        # 1. HuggingFace FlaxPreTrainedModel: model(**inputs, params=params) → output
        # 2. Flax Linen modules: model.apply({"params": params}, **inputs)
        # 3. Raw array input (e.g. AlexNet): passed positionally with train=False
        # 4. EasyDel / Flax NNX: nnx.split → (graphdef, state); forward via nnx.merge
        flax_params = params

        try:
            import flax.nnx as _nnx
        except ImportError:
            _nnx = None

        from transformers.modeling_flax_utils import FlaxPreTrainedModel  # noqa
        if isinstance(model, FlaxPreTrainedModel):
            def forward(params, inputs):
                out = model(**inputs, params=params, train=False)
                if hasattr(out, "logits") and out.logits is not None:
                    return out.logits
                if isinstance(out, (tuple, list)):
                    return out[0]
                return out
        elif hasattr(model, 'apply'):
            # Flax Linen .apply() style. Supports two input conventions:
            # - dict inputs (e.g. {"pixel_values": ...}): unpacked as kwargs
            # - raw array inputs (e.g. AlexNet): passed positionally with train=False
            #
            # params may be the full variable collection {"params": ..., "batch_stats": ...}
            # (produced by eval_shape init for Linen models with BatchNorm) or just the
            # raw params subtree (from from_pretrained). Detect by checking for the
            # "params" key at top level — HF params trees use model-specific keys.
            def forward(params, inputs):
                var_coll = params if (isinstance(params, dict) and "params" in params) else {"params": params}
                if isinstance(inputs, _Mapping):
                    out = model.apply(var_coll, **inputs)
                else:
                    out = model.apply(var_coll, inputs, train=False)
                if hasattr(out, "logits") and out.logits is not None:
                    return out.logits
                if isinstance(out, (tuple, list)):
                    return out[0]
                return out
        elif _nnx is not None and isinstance(model, _nnx.Module):
            # EasyDel / Flax NNX model.  EasyDel bakes a 5D internal mesh
            # (dp/fsdp/tp/sp/expert) into every model; TT MLIR only supports
            # 1D/2D mesh, so any with_sharding_constraint referencing that mesh
            # crashes the StableHLO pipeline.
            #
            # Strategy: re-load the equivalent native HuggingFace FlaxPreTrainedModel
            # from the same pretrained checkpoint (already in HF cache from EasyDel's
            # download).  The HF model has no sharding infrastructure and goes through
            # the Type A (FlaxPreTrainedModel) path cleanly.
            # EasyDel doesn't propagate _name_or_path to config; fall back to the
            # name we stashed on the model by loader() via _compiletron_pretrained_name.
            _pretrained_name = (
                getattr(model, "_compiletron_pretrained_name", None)
                or getattr(getattr(model, "config", None), "_name_or_path", None)
                or getattr(getattr(model, "config", None), "name_or_path", None)
            ) or None
            _hf_fallback_ok = False
            if _pretrained_name:
                try:
                    from transformers import FlaxAutoModelForCausalLM as _FlaxAutoCLM
                    _print_live_info(
                        f"EasyDel NNX detected — loading native Flax model to bypass 5D mesh"
                    )
                    # _do_init=False avoids TT eager init (SliceOp limitation) and
                    # returns a (model, params) tuple instead of a model-only object.
                    _hf_result = _FlaxAutoCLM.from_pretrained(_pretrained_name, _do_init=False)
                    if isinstance(_hf_result, tuple):
                        _hf_model, _hf_params = _hf_result
                    else:
                        _hf_model, _hf_params = _hf_result, _hf_result.params
                    # Swap out model/params for the HF Flax version (Type A path)
                    model = _hf_model
                    params = _hf_params
                    flax_params = params
                    _hf_fallback_ok = True

                    def forward(params, inputs):
                        out = model(**inputs, params=params, train=False)
                        if hasattr(out, "logits") and out.logits is not None:
                            return out.logits
                        if isinstance(out, (tuple, list)):
                            return out[0]
                        return out

                except Exception as _fe:
                    _print_live_info(
                        f"HF Flax fallback failed ({type(_fe).__name__}: {_fe!s:.80}) — "
                        "falling back to nnx.split/merge path"
                    )

            if not _hf_fallback_ok:
                # Last resort: nnx.split/merge.  Patch eformer.escale sharding
                # constraint (used by EasyDel) to a no-op during tracing.
                import eformer.escale as _escale
                _graphdef, _nnx_state = _nnx.split(model)
                flax_params = _nnx_state
                _nnx_mesh = Mesh(np.array([device]), ("batch",))
                _orig_als = _escale.with_sharding_constraint

                def forward(params, inputs):
                    _escale.with_sharding_constraint = lambda x, *_a, **_kw: x
                    try:
                        _m = _nnx.merge(_graphdef, params)
                        if isinstance(inputs, _Mapping):
                            out = _m(**inputs)
                        else:
                            out = _m(inputs)
                    finally:
                        _escale.with_sharding_constraint = _orig_als
                    if hasattr(out, "logits") and out.logits is not None:
                        return out.logits
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out
        else:
            raise ValueError(
                f"{type(model).__name__} has no .apply() method and is not a Flax NNX module — "
                "XLA worker requires HuggingFace Flax, Flax Linen, or EasyDel NNX models."
            )

        # _nnx_mesh is only set for EasyDel (Type C) models; None otherwise.
        _nnx_mesh = locals().get("_nnx_mesh", None)

        if mesh_chips > 1:
            # ── Multi-chip path ───────────────────────────────────────────────
            # Two sub-paths based on loader type (from _LoaderMeta):
            #   Type A (use_shard_map=True):  shard_map with named axis "X"
            #   Type B (use_shard_map=False): jax.jit(in_shardings=...) data-parallel
            # Both achieve data parallelism; shard_map positions Type A for future
            # tensor-parallel work via load_multichip_model().
            all_devices = jax.devices()
            n = min(mesh_chips, len(all_devices))
            if n < mesh_chips:
                _print_live_info(
                    f"Only {n} TT device(s) visible — using {n}-chip data-parallel"
                )

            use_shard_map = meta is not None and meta.use_shard_map
            axis = meta.axis_name if meta is not None else "batch"

            if use_shard_map:
                # ── Type A: Linen model — shard_map data-parallel ────────────
                # Uses shard_map instead of jit(in_shardings=...) to provide a
                # named collective axis context matching the loader's axis_name="X".
                # Both paths achieve data-parallel; shard_map positions this code
                # for future tensor-parallel work via load_multichip_model().
                from jax.experimental.shard_map import shard_map as _shard_map

                mesh       = Mesh(np.array(all_devices[:n]), axis_names=(axis,))
                replicated = NamedSharding(mesh, PartitionSpec())
                sharded    = NamedSharding(mesh, PartitionSpec(axis,))

                _print_live_info(
                    f"★ {n}-CHIP SHARD_MAP ACTIVE (Type A, axis={axis!r}): "
                    f"{[str(d) for d in all_devices[:n]]} "
                    f"(batch={n}, one example per chip)"
                )

                single = make_input(all_devices[0])
                # BatchEncoding / FeatureExtraction inherit UserDict but are not
                # registered JAX pytrees; convert to plain dict for jax.tree_util.
                if isinstance(single, _Mapping) and type(single) is not dict:
                    single = dict(single)
                if isinstance(single, _Mapping):
                    dummy_inputs = {k: jnp.concatenate([v] * n, axis=0)
                                    for k, v in single.items()}
                else:
                    dummy_inputs = jnp.concatenate([single] * n, axis=0)

                sharded_params = jax.device_put(flax_params, replicated)
                sharded_inputs = jax.device_put(dummy_inputs, sharded)

                compiled_fn = jax.jit(_shard_map(
                    forward,
                    mesh=mesh,
                    in_specs=(PartitionSpec(), PartitionSpec(axis)),
                    out_specs=PartitionSpec(axis),
                    check_rep=False,
                ))

                _print_live_info(
                    f"Input shape: {jax.tree_util.tree_map(lambda x: x.shape, dummy_inputs)}"
                )
                _print_progress_step(2, 3, f"Compiling via shard_map across {n} chips (Type A)...")
                compile_start = time.time()

                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    output = compiled_fn(sharded_params, sharded_inputs)
                    output.block_until_ready()
                    signal.alarm(0)
                except TimeoutException:
                    signal.alarm(0)
                    return False, None, time.time() - compile_start, 0.0, "TIMEOUT", None

                compile_time = time.time() - compile_start
                _print_progress_step(3, 3,
                    f"Output shape: {output.shape}  ({compile_time:.1f}s — {n} chips shard_map)")

                infer_s = 0.0
                try:
                    infer_start = time.time()
                    _ = compiled_fn(sharded_params, sharded_inputs)
                    _.block_until_ready()
                    infer_s = time.time() - infer_start
                except Exception:
                    pass

                return True, output, compile_time, infer_s, "", (compiled_fn, sharded_params, all_devices[0], (n, sharded_inputs))

            else:
                # ── Type B: existing data-parallel jit path ──────────────────
                # Params are replicated across all N chips; the input batch (size N)
                # is sharded so each chip processes exactly one example.  XLA JIT
                # generates a single program that runs on all chips in parallel.
                mesh       = Mesh(np.array(all_devices[:n]), axis_names=(axis,))
                replicated = NamedSharding(mesh, PartitionSpec())
                batched    = NamedSharding(mesh, PartitionSpec(axis,))

                _print_live_info(
                    f"★ {n}-CHIP MESH ACTIVE (Type B, axis={axis!r}): {[str(d) for d in all_devices[:n]]} "
                    f"(batch={n}, one example per chip)"
                )

                sharded_params = jax.device_put(flax_params, replicated)

                # Build a batch of n identical inputs (one element per device).
                # BatchEncoding / FeatureExtraction are UserDict subclasses that are
                # NOT registered JAX pytrees; convert to plain dict first.
                single = make_input(all_devices[0])
                if isinstance(single, _Mapping) and type(single) is not dict:
                    single = dict(single)
                if isinstance(single, _Mapping):
                    dummy_inputs = {k: jnp.concatenate([v] * n, axis=0) for k, v in single.items()}
                else:
                    dummy_inputs = jnp.concatenate([single] * n, axis=0)
                sharded_inputs = jax.device_put(dummy_inputs, batched)

                compiled_fn = jax.jit(
                    forward,
                    in_shardings=(replicated, batched),
                    out_shardings=batched,
                )

                _print_live_info(
                    f"Input shape: {jax.tree_util.tree_map(lambda x: x.shape, dummy_inputs)}"
                )
                _print_progress_step(2, 3, f"Compiling via JAX JIT across {n} chips (data-parallel)...")
                compile_start = time.time()

                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    output = compiled_fn(sharded_params, sharded_inputs)
                    output.block_until_ready()
                    signal.alarm(0)
                except TimeoutException:
                    signal.alarm(0)
                    return False, None, time.time() - compile_start, 0.0, "TIMEOUT", None

                compile_time = time.time() - compile_start
                _print_progress_step(3, 3, f"Output shape: {output.shape}  ({compile_time:.1f}s — {n} chips data-parallel)")

                # Second pass for pure infer_s measurement (JIT already done).
                infer_s = 0.0
                try:
                    infer_start = time.time()
                    _ = compiled_fn(sharded_params, sharded_inputs)
                    _.block_until_ready()
                    infer_s = time.time() - infer_start
                except Exception:
                    pass

                # Return the lead device's bundle for optional First Voice and bench passes.
                return True, output, compile_time, infer_s, "", (compiled_fn, sharded_params, all_devices[0], (n, sharded_inputs))

        else:
            # ── Single-chip path ─────────────────────────────────────────────
            compiled_fn = jax.jit(forward)

            dummy_inputs = make_input(device)
            # BatchEncoding / FeatureExtraction are not registered JAX pytrees;
            # convert to plain dict so tree_map and JIT can traverse them.
            if isinstance(dummy_inputs, _Mapping) and type(dummy_inputs) is not dict:
                dummy_inputs = dict(dummy_inputs)
            _print_live_info(f"Input shape: {jax.tree_util.tree_map(lambda x: x.shape, dummy_inputs)}")

            _print_progress_step(2, 3, "Compiling via JAX JIT (first call triggers XLA)...")
            compile_start = time.time()

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                with jax.default_device(device):
                    if _nnx_mesh is not None:
                        with _nnx_mesh:
                            output = compiled_fn(flax_params, dummy_inputs)
                            output.block_until_ready()
                    else:
                        output = compiled_fn(flax_params, dummy_inputs)
                        output.block_until_ready()  # ensure device execution completes
                signal.alarm(0)
            except TimeoutException:
                signal.alarm(0)
                return False, None, time.time() - compile_start, 0.0, "TIMEOUT", None

            compile_time = time.time() - compile_start
            _print_progress_step(3, 3, f"Output shape: {output.shape}  ({compile_time:.1f}s)")

            # Second pass for pure infer_s measurement (JIT already done).
            infer_s = 0.0
            try:
                with jax.default_device(device):
                    if _nnx_mesh is not None:
                        with _nnx_mesh:
                            infer_start = time.time()
                            _ = compiled_fn(flax_params, dummy_inputs)
                            _.block_until_ready()
                            infer_s = time.time() - infer_start
                    else:
                        infer_start = time.time()
                        _ = compiled_fn(flax_params, dummy_inputs)
                        _.block_until_ready()
                        infer_s = time.time() - infer_start
            except Exception:
                pass

            return True, output, compile_time, infer_s, "", (compiled_fn, flax_params, device, (1, dummy_inputs))

    except TimeoutException:
        signal.alarm(0)
        return False, None, 0.0, 0.0, "TIMEOUT", None
    except Exception as e:
        signal.alarm(0)
        _emsg = str(e)
        if not _emsg:
            import traceback as _tb
            _emsg = "".join(_tb.format_exception(type(e), e, e.__traceback__)).strip()[-300:]
        return False, None, 0.0, 0.0, f"{type(e).__name__}: {_emsg[:300]}", None


def _attempt_first_voice_xla(
    compiled_bundle,
    task: str,
    model_id: str,
    tokenizer=None,
    timeout: int = 60,
) -> tuple[str, dict | None]:
    """Run a themed First Voice pass using the XLA compiled function.

    compiled_bundle is (compiled_fn, params, device, ...) returned by _compile_model_xla.
    """
    if compiled_bundle is None:
        return "", None

    compiled_fn, params, device = compiled_bundle[:3]

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from lib.expedition.sampler import get_sample

        sample = get_sample(task)
        if sample is None:
            return "", None

        # Build JAX input from the sample.
        if sample["input_type"] == "text":
            if tokenizer is None:
                return "", None
            text = sample["data"]
            if isinstance(text, dict):
                text = text.get("context", "") + " " + text.get("question", "")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            enc = tokenizer(str(text), return_tensors="np",
                            max_length=32, padding="max_length", truncation=True)
            inputs = {k: jnp.array(v) for k, v in enc.items()
                      if k in ("input_ids", "attention_mask")}
        elif sample["input_type"] == "image":
            from PIL import Image
            import numpy as np
            img = Image.open(sample["data"]).convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            inputs = {"pixel_values": jnp.array(arr[None].transpose(0, 3, 1, 2))}
        else:
            return "", None

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            with jax.default_device(device):
                output = compiled_fn(params, inputs)
                output.block_until_ready()
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            return "", None

        # Decode: top-3 next-token predictions for text tasks.
        import numpy as np
        arr = np.array(output)
        if arr.ndim == 3:   # (batch, seq, vocab)
            last = arr[0, -1, :]
        elif arr.ndim == 2: # (batch, vocab)
            last = arr[0, :]
        else:
            return "", None

        if tokenizer is not None and last.shape[-1] > 1:
            top_k = min(5, last.shape[-1])
            indices = last.argsort()[-top_k:][::-1]
            probs = _softmax(last)[indices]
            words = []
            for idx, p in zip(indices.tolist(), probs.tolist()):
                w = tokenizer.decode([idx], skip_special_tokens=True).strip()
                if w:
                    words.append(f"{w} ({p:.0%})")
            if words:
                return "→ " + " | ".join(words[:3]), sample

        return "", None

    except Exception:
        signal.alarm(0)
        return "", None


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Queue / loader ────────────────────────────────────────────────────────────

@dataclass
class QueueItem:
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


@dataclass
class _LoaderMeta:
    """Routing metadata returned alongside a loader callable from _build_loader_xla.

    use_shard_map=True: Type A Linen loader (has load_multichip_model).
      _compile_model_xla uses jax.experimental.shard_map with axis_name.
    use_shard_map=False: Type B EasyDel/NNX loader (or single-chip).
      _compile_model_xla uses jax.jit(in_shardings=...) data-parallel path.
    axis_name: mesh axis label. "X" for Type A, "batch" for Type B.

    TODO: when CPU mesh init is available, Type A should call load_multichip_model()
    for the tensor-parallel Linen module instead of load_model().
    """
    use_shard_map: bool = False
    axis_name: str = "batch"


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


def _load_single_model_xla(model_json_path: str) -> QueueItem:
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


def _build_loader_xla(item: QueueItem):
    """Return a callable → (model, params, tokenizer, make_input_fn) for XLA.

    For seed models, imports the JAX loader from tt-forge-models.
    For frontier models, uses FlaxAutoModelForCausalLM from transformers.

    make_input_fn(device) → dict of jax.Arrays (the dummy input for JIT trace).
    """
    import jax
    import jax.numpy as jnp

    forge_models_path = os.path.expanduser("~/code/tt-forge-models")

    if item.is_frontier:
        # HuggingFace frontier: use transformers Flax auto-classes.
        from transformers import (
            FlaxAutoModelForCausalLM,
            FlaxAutoModelForSeq2SeqLM,
            FlaxAutoModelForMaskedLM,
            FlaxAutoModelForImageClassification,
            AutoTokenizer,
            AutoConfig,
        )

        _FLAX_CLASS = {
            "text-generation":      FlaxAutoModelForCausalLM,
            "text2text-generation": FlaxAutoModelForSeq2SeqLM,
            "fill-mask":            FlaxAutoModelForMaskedLM,
            "image-classification": FlaxAutoModelForImageClassification,
        }
        flax_cls = _FLAX_CLASS.get(item.task, FlaxAutoModelForCausalLM)

        def loader():
            # Use _do_init=False to skip Flax's eager init which fails on TT
            # due to SliceOp limitations in pjrt-plugin-tt 0.9.0. JIT inference
            # compiles SliceOps via XLA which works. Returns (model, params) tuple.
            result = flax_cls.from_pretrained(item.model_id, dtype="float32", _do_init=False)
            if isinstance(result, tuple):
                model, params = result
            else:
                model, params = result, result.params
            # Reattach params so model.params doesn't raise ValueError for internal
            # accesses. Encoder-decoder models (blenderbot, BART) read self.params
            # inside their forward pass even when params is also passed as a kwarg.
            # In older transformers, the params setter is absent; bypass the property
            # by writing _params directly so the getter finds real weights.
            try:
                model.params = params
            except Exception:
                try:
                    object.__setattr__(model, '_params', params)
                except Exception:
                    pass

            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    item.model_id, trust_remote_code=True
                )
            except Exception:
                pass

            if item.task in ("text-generation", "text2text-generation", "fill-mask"):
                def make_input(device):
                    seq = 32
                    return {"input_ids": jnp.ones((1, seq), dtype=jnp.int32)}
            else:
                def make_input(device):
                    return {"pixel_values": jnp.zeros((1, 3, 224, 224), dtype=jnp.float32)}

            return model, params, tokenizer, make_input

        return loader, _LoaderMeta()

    else:
        # Seed model from tt-forge-models JAX loader.
        if item.loader_module is None or item.loader_class is None:
            raise ValueError(
                f"Non-frontier XLA model {item.model_id!r} missing loader_module/loader_class"
            )
        import importlib, types
        # Register synthetic _forgems root package so relative imports in the
        # loaders (e.g. `from ....base import ForgeModel`) resolve correctly.
        _PKG = "_forgems"
        if _PKG not in sys.modules:
            root_mod = types.ModuleType(_PKG)
            root_mod.__path__ = [forge_models_path]
            root_mod.__package__ = _PKG
            root_mod.__file__ = os.path.join(forge_models_path, "__init__.py")
            sys.modules[_PKG] = root_mod

        mod = importlib.import_module(item.loader_module)
        cls = getattr(mod, item.loader_class)

        # Parse variant from display_name if loader class supports it.
        variant_str = item.display_name.split()[-1] if item.display_name else None
        try:
            variant_enum = getattr(mod, "ModelVariant", None)
            variant = variant_enum(variant_str) if variant_enum and variant_str else None
        except Exception:
            variant = None

        instance = cls(variant=variant) if variant is not None else cls()

        # Detect Type A: has load_multichip_model (Linen tensor-parallel model).
        # When mesh_chips > 1, use shard_map path in _compile_model_xla.
        # NOTE: the loader still calls load_model() (not load_multichip_model()) because
        # initializing the multichip Linen module requires a CPU mesh context not
        # available in the TT XLA worker. True tensor-parallelism is a follow-on task.
        _is_type_a = (
            item.mesh_chips > 1
            and hasattr(instance, "load_multichip_model")
        )
        meta = _LoaderMeta(use_shard_map=True, axis_name="X") if _is_type_a else _LoaderMeta()

        def loader():
            import jax.numpy as jnp

            # Patch FlaxPreTrainedModel.from_pretrained to default _do_init=False.
            # Without this the seed loaders' from_pretrained runs eager Flax init
            # which hits SliceOp failures on TT hardware (XlaRuntimeError code 13).
            # With _do_init=False, from_pretrained returns a (model, params) tuple
            # instead of a model with .params — handle both shapes below.
            #
            # Two exceptions where _do_init=False must be skipped:
            # 1. from_pt=True — HuggingFace accesses model.params internally during
            #    PyTorch→Flax weight conversion; from_pt already avoids TT eager init.
            # 2. Models with no native Flax weights (e.g. DINOv2) — from_pretrained
            #    triggers an internal fallback that calls model.params. Retry with
            #    from_pt=True which forces the PyTorch→Flax conversion path instead.
            _FPTM = None
            _orig = None
            try:
                from transformers.modeling_flax_utils import FlaxPreTrainedModel as _FPTM
                _orig = _FPTM.from_pretrained
                _orig_func = _orig.__func__
                @classmethod  # type: ignore[misc]
                def _patched(cls, *args, **kw):
                    if kw.get("from_pt", False):
                        # from_pt=True handles weight init without TT eager execution;
                        # adding _do_init=False would break the internal conversion.
                        return _orig_func(cls, *args, **kw)
                    kw.setdefault("_do_init", False)
                    try:
                        return _orig_func(cls, *args, **kw)
                    except ValueError as _ve:
                        if "params" in str(_ve) and "_do_init" in str(_ve):
                            # No native Flax weights — retry via PyTorch→Flax conversion.
                            kw.pop("_do_init", None)
                            kw["from_pt"] = True
                            return _orig_func(cls, *args, **kw)
                        raise
                _FPTM.from_pretrained = _patched
            except Exception:
                pass

            try:
                result = instance.load_model()
            finally:
                if _FPTM is not None and _orig_func is not None:
                    # Restore as a proper classmethod descriptor, NOT the bound method
                    # captured in _orig.  Restoring a bound method bakes cls=FlaxPreTrainedModel
                    # into the attribute, breaking subclass MRO dispatch (FlaxGPT2LMHeadModel
                    # would call from_pretrained with cls=FlaxPreTrainedModel → missing 'module'
                    # TypeError when the HF Flax fallback runs later in _compile_model_xla).
                    _FPTM.from_pretrained = classmethod(_orig_func)

            if isinstance(result, tuple):
                model, params = result
            else:
                model = result
                try:
                    params = model.params
                except Exception:
                    params = {}
            # Stash the pretrained model name on the model object so _compile_model_xla
            # can find it for the EasyDel NNX → HF Flax fallback path.  EasyDel doesn't
            # propagate _name_or_path into its config, so we carry it ourselves.
            try:
                _pname = getattr(instance, "_model_name", None) or getattr(instance, "model_name", None)
                if _pname:
                    object.__setattr__(model, "_compiletron_pretrained_name", _pname)
            except Exception:
                pass
            # Reattach so model.params doesn't raise ValueError for internal accesses.
            # Use object.__setattr__ as a fallback when the property setter is absent
            # (older transformers builds) — bypasses the property directly to _params.
            try:
                model.params = params
            except Exception:
                try:
                    object.__setattr__(model, '_params', params)
                except Exception:
                    pass

            # For custom Linen models (e.g. AlexNet) that initialize parameters
            # with a random key instead of from_pretrained, params will be empty
            # here. Fall back to load_parameters() which runs model.init().
            if not params and hasattr(instance, "load_parameters"):
                try:
                    params = instance.load_parameters()
                except Exception:
                    pass

            tokenizer = instance._load_tokenizer() if hasattr(instance, "_load_tokenizer") else None

            # Prefer the loader's own load_inputs() for sample data — custom Linen
            # models (e.g. AlexNet) use positional raw arrays, not HF-style dicts.
            # Fall back to task-name heuristics for loaders without load_inputs().

            # Inject a longer sample text so text models compile/bench with a
            # realistic sequence length instead of whatever short string the loader
            # uses by default (e.g. BLOOM's 4-token "Hello there fellow traveler").
            if hasattr(instance, "sample_text"):
                instance.sample_text = _BENCH_SAMPLE_TEXT

            _sample_inputs = None
            if hasattr(instance, "load_inputs"):
                try:
                    _sample_inputs = instance.load_inputs()
                except Exception:
                    pass

            # Pad short text sequences up to _BENCH_SEQ_LEN.  Only applies to
            # 2-D dict inputs with an input_ids key; image/audio shapes are left alone.
            if (
                _sample_inputs is not None
                and isinstance(_sample_inputs, _Mapping)
                and "input_ids" in _sample_inputs
            ):
                _ids = _sample_inputs["input_ids"]
                if hasattr(_ids, "shape") and _ids.ndim == 2 and _ids.shape[1] < _BENCH_SEQ_LEN:
                    _pad_len = _BENCH_SEQ_LEN - _ids.shape[1]
                    _padded = {}
                    for _k, _v in _sample_inputs.items():
                        if hasattr(_v, "shape") and _v.ndim == 2 and _v.shape[1] == _ids.shape[1]:
                            # attention_mask: 0 = ignore padding; input_ids: 1 (generic pad id)
                            _pval = 0 if _k == "attention_mask" else 1
                            _pad = jnp.full((_v.shape[0], _pad_len), _pval, dtype=_v.dtype)
                            _padded[_k] = jnp.concatenate([_v, _pad], axis=1)
                        else:
                            _padded[_k] = _v
                    _sample_inputs = _padded

            if _sample_inputs is not None:
                def make_input(device, _si=_sample_inputs):
                    return _si
            else:
                # Fallback: infer from task name.
                task_lower = item.task.lower()
                if "image" in task_lower or "vision" in task_lower or "classification" in task_lower:
                    def make_input(device):
                        return {"pixel_values": jnp.zeros((1, 3, 224, 224), dtype=jnp.float32)}
                elif "audio" in task_lower or "speech" in task_lower:
                    def make_input(device):
                        return {"input_features": jnp.zeros((1, 80, 3000), dtype=jnp.float32)}
                else:
                    def make_input(device):
                        return {"input_ids": jnp.ones((1, 32), dtype=jnp.int32)}

            return model, params, tokenizer, make_input

        return loader, meta


# ── Main worker loop ──────────────────────────────────────────────────────────

def run_worker_xla(chip_id: int, run_number: int, bestiary_path: str,
                   queue_path: str | None, results_path: str,
                   model_json_path: str | None = None,
                   bench_passes: int = 0,
                   bench_shapes: bool = False,
                   ephemeral: bool = False,
                   evict_failures: bool = False,
                   run_total: int = 0,
                   no_permafail: bool = False) -> None:
    """Main entry point for the XLA per-chip worker.

    Same interface as expedition_worker.run_worker but uses JAX/Flax instead
    of forge/PyTorch. Results CSV includes backend="xla" for bestiary queries.

    Args:
        chip_id:         Zero-based index of the TT chip this worker owns.
        run_number:      Sequential expedition run number.
        bestiary_path:   Path to the bestiary JSON file (created if absent).
        queue_path:      Path to the queue JSON file for this chip. Optional when
                         model_json_path is provided.
        results_path:    Path to write the per-chip CSV results file. Opens in append
                         mode so multiple per-model invocations accumulate results.
        model_json_path: Optional path to a single-model JSON file (a flat dict rather
                         than a list). When provided, overrides queue_path and processes
                         exactly one model. The TUI uses this for per-model dispatch.
        bench_passes:    Number of timed inference passes to run after each successful
                         compile (2 warm-up first).  0 disables benchmarking.
        bench_shapes:    Accepted for CLI compatibility with the forge worker; XLA ignores
                         this flag because shape sweeping is not implemented for JAX.
        ephemeral:       If True, evict net-new HF downloads after each model unless gold star.
        evict_failures:  If True and ephemeral, also evict weights for failed models.
    """
    import datetime
    from lib.expedition.bestiary import Bestiary
    from lib.expedition.hud import ChipHUD
    from lib.expedition.scorer import (
        compute_rarity, compute_newness, compute_score, Rarity, Newness,
    )

    # ── JAX init ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{TEAL}{'═'*80}{RESET}")
    print(f"{BOLD}{TEAL}  EXPEDITION XLA CHIP {chip_id}  ·  run #{run_number:03d}{RESET}")
    print(f"{BOLD}{TEAL}{'═'*80}{RESET}\n")

    try:
        device = _setup_jax(chip_id)
    except Exception as e:
        err_short = str(e)[:200]
        print(f"{RED}✗ JAX/TT device init failed: {err_short}{RESET}")
        print(f"{DIM}  Is pjrt-plugin-tt installed in this Python environment?{RESET}")
        print(f"{DIM}  Run: xla-venv/bin/pip show pjrt-plugin-tt{RESET}")
        # Write failure rows for every model in this dispatch so the TUI gets
        # a proper result instead of silence (which leaves the chip stuck).
        try:
            if model_json_path:
                _fail_items = [_load_single_model_xla(model_json_path)]
            elif queue_path:
                _fail_items = _load_queue(queue_path)
            else:
                _fail_items = []
            if _fail_items:
                Path(results_path).parent.mkdir(parents=True, exist_ok=True)
                _empty = not Path(results_path).exists() or Path(results_path).stat().st_size == 0
                with open(results_path, "a", newline="") as _f:
                    _w = csv.DictWriter(_f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
                    if _empty:
                        _w.writeheader()
                    for _it in _fail_items:
                        _w.writerow({
                            "model": _it.model_id, "status": "failed",
                            "error": f"XLA init failed: {err_short}", "pts": -10,
                            "backend": BACKEND_LABEL,
                        })
        except Exception:
            pass
        sys.exit(1)

    bestiary = Bestiary(path=bestiary_path)
    if model_json_path:
        # Per-model TUI dispatch: process a single model from a flat JSON dict.
        queue = [_load_single_model_xla(model_json_path)]
    elif queue_path:
        # Normal batch dispatch: process the whole chip queue JSON (list of dicts).
        queue = _load_queue(queue_path)
    else:
        raise ValueError("Either queue_path or model_json_path must be provided")
    hud = ChipHUD(chip_id=chip_id, total_models=run_total or len(queue), run_number=run_number)
    hud.write_status()

    _set_pane_title(f"C{chip_id}·XLA · {len(queue)} queued · run #{run_number:03d}")
    print(f"  {len(queue)} models queued on {device}\n")

    results: list[dict] = []
    last_artifact = ""
    max_mesh_chips = 1  # track highest chip-count seen; drives closing banner

    # ── Ephemeral cache setup ────────────────────────────────────────────────
    # Snapshot which HF cache entries already exist before the run so that
    # maybe_evict() can distinguish net-new downloads from pre-existing ones.
    preexisting: frozenset[str] = frozenset()
    if ephemeral:
        from lib.expedition import cache_janitor as _janitor
        preexisting = _janitor.snapshot_preexisting()

    for idx, item in enumerate(queue, 1):
        hud.set_current(item.model_id, idx)
        hud.write_status()
        s = hud.state
        _xla_parts = item.model_id.split("/")
        _xla_be = {"pytorch","jax","onnx","tensorflow","flax","paddle","paddlepaddle"}
        short_name = (_xla_parts[0] if _xla_parts[-1].lower() in _xla_be else _xla_parts[-1])[:24]
        _set_pane_title(
            f"C{chip_id}·XLA [{idx}/{s.total_models}] {short_name}"
            f"  ✓{s.successes} ✗{s.failures}  {s.pts}pts"
        )

        max_mesh_chips = max(max_mesh_chips, item.mesh_chips)
        is_first_ever = not bestiary.is_compiled(item.model_id)
        rarity = compute_rarity(item.hf_downloads)
        newness = compute_newness(item.hf_created_at, is_first_ever)

        _print_rarity_reveal(
            model_id=item.model_id, rarity=rarity.value, newness=newness.value,
            task=item.task, source=item.source, is_first_ever=is_first_ever,
        )

        if last_artifact:
            print(f"  {DIM}last: {last_artifact[:80]}{RESET}")

        _print_progress_step(1, 3, "Loading Flax model...")
        start = time.time()

        # ── Loader construction ──────────────────────────────────────────────
        try:
            loader, _meta = _build_loader_xla(item)
        except Exception as e:
            elapsed = time.time() - start
            _print_failure(item.model_id, str(e), elapsed)
            score = compute_score(False, is_first_ever, rarity, newness,
                                  hud.state.streak, mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            bestiary.record_failure(item.model_id, run_number, str(e))
            hud.write_status()
            results.append({"model": item.model_id, "status": "failed",
                            "error": str(e), "pts": score.pts, "backend": BACKEND_LABEL})
            continue

        # ── Compile + inference ──────────────────────────────────────────────
        success, output, compile_time, infer_time, error_str, compiled_bundle = _compile_model_xla(
            loader, device, chip_id, mesh_chips=item.mesh_chips, meta=_meta
        )
        # Auto-install missing packages and retry once
        if not success and "No module named" in error_str:
            if _try_install_missing(error_str):
                success, output, compile_time, infer_time, error_str, compiled_bundle = _compile_model_xla(
                    loader, device, chip_id, mesh_chips=item.mesh_chips, meta=_meta
                )
        elapsed = time.time() - start

        if success:
            # Decode: convert JAX array to a human-readable artifact string.
            import numpy as np
            arr = np.array(output)
            if arr.ndim == 3:
                artifact = f"shape={arr.shape}  max={arr.max():.3f}  mean={arr.mean():.3f}"
            elif arr.ndim == 2:
                artifact = f"shape={arr.shape}  max={arr.max():.3f}"
            else:
                artifact = f"shape={arr.shape}"

            last_artifact = artifact

            # ── First Voice ──────────────────────────────────────────────────
            tokenizer = None
            if compiled_bundle is not None:
                # Try to extract tokenizer from loader result for First Voice.
                try:
                    _, _, tokenizer, _ = loader()
                except Exception:
                    pass

            first_voice_text, first_voice_sample = _attempt_first_voice_xla(
                compiled_bundle=compiled_bundle,
                task=item.task,
                model_id=item.model_id,
                tokenizer=tokenizer,
            )
            is_first_voice = bool(first_voice_text)

            # Compute throughput from the second inference pass (infer_time).
            throughput, throughput_unit = _compute_throughput_xla(
                item.task, output, infer_time, batch=item.mesh_chips
            )

            score = compute_score(success=True, is_first_ever=is_first_ever,
                                  rarity=rarity, newness=newness,
                                  streak=hud.state.streak,
                                  mesh_chips=item.mesh_chips,
                                  is_first_voice=is_first_voice)
            hud.record_success(item.model_id, score)

            # Run bench passes if requested (warm-up + timed) using compiled bundle.
            bench_stats: dict = {}
            if bench_passes > 0 and compiled_bundle is not None:
                bench_stats = _run_bench_passes_xla(compiled_bundle, item.task, bench_passes)

            _print_success(
                item.model_id, compile_time, infer_time, artifact,
                score.pts, is_first_ever, hud.state.streak,
                throughput=throughput, throughput_unit=throughput_unit,
                bench_p50=bench_stats.get("infer_p50_s", 0.0),
            )

            if is_first_voice and first_voice_sample:
                print(f"    {GOLD}🗣 First Voice{RESET}  "
                      f"{DIM}[{first_voice_sample['description']}]{RESET}")
                print(f"    {PINK}{first_voice_text[:120]}{RESET}")

            compiled_at = datetime.datetime.now().isoformat()

            # Journal entry for First Voice results.
            if is_first_voice and first_voice_sample:
                try:
                    from lib.expedition.notes import journal_entry
                    project_dir = Path(__file__).resolve().parent.parent.parent
                    journal_entry(
                        run_number=run_number, chip_id=chip_id,
                        model_id=item.model_id, task=item.task,
                        sample_description=first_voice_sample["description"],
                        first_voice_text=first_voice_text,
                        compile_time_s=compile_time, score_pts=score.pts,
                        project_dir=project_dir,
                    )
                except Exception:
                    pass

            bestiary.save_artifact(
                model_id=item.model_id, task=item.task, compiled_at=compiled_at,
                chip=chip_id, run=run_number,
                artifact_text=first_voice_text if is_first_voice else artifact,
            )
            bestiary.record_success(
                model_id=item.model_id, chip=chip_id, run=run_number,
                time_s=compile_time, task=item.task, source=item.source,
                rarity=rarity.value, hf_downloads=item.hf_downloads,
                hf_created_at=item.hf_created_at,
                artifact=first_voice_text if is_first_voice else artifact,
                backend=BACKEND_LABEL,
                first_voice=first_voice_text if is_first_voice else "",
                compile_s=compile_time,
                infer_s=infer_time,
                throughput=throughput,
                throughput_unit=throughput_unit,
                mesh_chips=item.mesh_chips,
            )
            bestiary.add_chip_points(chip=chip_id, pts=score.pts,
                                     first_ever=is_first_ever, streak=hud.state.streak)

            # Write split timing and bench stats to the perf log.
            perf_record: dict = {
                "model_id":        item.model_id,
                "run":             run_number,
                "timestamp":       compiled_at,
                "backend":         BACKEND_LABEL,
                "chip":            chip_id,
                "compile_s":       round(compile_time, 4),
                "infer_s":         round(infer_time, 4),
                "throughput":      round(throughput, 3),
                "throughput_unit": throughput_unit,
            }
            perf_record.update(bench_stats)
            bestiary.append_perf_record(perf_record)

            results.append({
                "model": item.model_id, "status": "success",
                "pts": score.pts, "compile_time": compile_time,
                "infer_time": infer_time,
                "artifact": first_voice_text if is_first_voice else artifact,
                "first_ever": is_first_ever, "first_voice": is_first_voice,
                "backend": BACKEND_LABEL,
            })
        else:
            _print_failure(item.model_id, error_str, elapsed)
            score = compute_score(False, is_first_ever, rarity, newness,
                                  hud.state.streak, mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            bestiary.record_failure(item.model_id, run_number, error_str)
            results.append({
                "model": item.model_id, "status": "failed",
                "error": error_str, "pts": score.pts, "backend": BACKEND_LABEL,
            })

        bestiary.save()

        # ── Ephemeral cache eviction ─────────────────────────────────────────
        # After persisting the result, optionally evict HF weights that were
        # downloaded for this model. Gold-star results are kept on disk as a
        # reward signal; failures are evicted only when evict_failures=True.
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
    _chip_label = (
        f"{max_mesh_chips}-CHIP RALLY" if max_mesh_chips > 1 else f"CHIP {chip_id}"
    )
    _set_pane_title(f"XLA {_chip_label} DONE  ✓{s.successes} ✗{s.failures}  {s.pts}pts")

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

    print(f"\n{BOLD}{TEAL}{'═'*80}{RESET}")
    print(f"{BOLD}XLA {_chip_label} DONE{RESET}  pts:{GOLD}{s.pts}{RESET}  "
          f"✓{s.successes} ✗{s.failures}  best streak: 🔥×{s.best_streak}")
    print(f"{BOLD}{TEAL}{'═'*80}{RESET}")
    try:
        input("Press Enter to close...")
    except EOFError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-chip XLA Expedition worker: JAX/Flax compile, decode, and score."
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
                        help="Sweep alternative input shapes (forge-only; XLA ignores this flag).")
    parser.add_argument("--ephemeral", action="store_true",
                        help="Evict net-new HF model weights after each result "
                             "unless the model earns a gold-star rating.")
    parser.add_argument("--evict-failures", action="store_true",
                        help="With --ephemeral, also evict weights for failed models.")
    parser.add_argument("--no-permafail", action="store_true",
                        help="Bypass the permanent-failure gate for this run.")
    args = parser.parse_args()
    if not args.queue and not args.model_json:
        parser.error("one of --queue or --model-json is required")
    if args.queue and args.model_json:
        parser.error("--queue and --model-json are mutually exclusive")

    run_worker_xla(
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
        no_permafail=args.no_permafail,
    )
