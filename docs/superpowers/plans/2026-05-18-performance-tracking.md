# Performance Tracking & Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split compile time from inference time, add tokens/sec + ms/sample throughput, write per-run perf_history.jsonl, and add optional inline bench passes with shape sweep.

**Architecture:** New fields added to `Bestiary.record_success()` capture split timings and throughput; `append_perf_record()` writes one JSONL line per model per run. `_compile_model` return tuple gains `infer_time` and `sample_inputs`. Pure helpers `_compute_throughput` and `_run_bench_passes` are unit-tested independently. Optional `--bench-passes N` and `--bench-shapes` CLI flags activate inline stress testing; both workers forward these flags.

**Tech Stack:** Python 3.11, stdlib only (`statistics`, `json`, `time`, `signal`). No new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `lib/expedition/bestiary.py` | Add `compile_s/infer_s/throughput/throughput_unit` params to `record_success()`; add `append_perf_record()` method |
| `lib/expedition/expedition_worker.py` | Extend `_compile_model` return to 7-tuple; add `_compute_throughput`, `_run_bench_passes`, `_run_shape_sweep`; update `_print_success`, `run_worker`, CSV fields, CLI args |
| `lib/expedition/expedition_worker_xla.py` | Add `infer_time` second-pass; add `_compute_throughput_xla`, `_run_bench_passes_xla`; update `_print_success`, `run_worker_xla`, CSV fields, CLI args |
| `lib/expedition/run_state.py` | Add `infer_time: float` field to `ModelResult`; update `from_csv_row()` |
| `expedition_tui.py` | Forward `--bench-passes` and `--bench-shapes` at both worker dispatch sites (lines ~1265 and ~1304) |
| `tests/lib/test_bestiary.py` | Add tests for new `record_success` fields and `append_perf_record` |
| `tests/lib/test_perf_worker_helpers.py` | New file: tests for `_compute_throughput` and `_run_bench_passes` |
| `tests/lib/test_run_state.py` | New file: tests for `ModelResult.infer_time` and `from_csv_row` |

---

## Task 1: Bestiary — perf summary fields + append_perf_record

**Files:**
- Modify: `lib/expedition/bestiary.py`
- Test: `tests/lib/test_bestiary.py`

- [ ] **Step 1: Write failing tests for new record_success fields**

Add to `tests/lib/test_bestiary.py`:

```python
class TestBestiaryPerfFields:
    def _make_success(self, b, model_id="m", compile_s=5.0, infer_s=0.5,
                      throughput=12.0, throughput_unit="tokens/sec"):
        b.record_success(
            model_id=model_id, chip=0, run=1, time_s=compile_s + infer_s,
            task="text-generation", source="hf", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="x",
            compile_s=compile_s, infer_s=infer_s,
            throughput=throughput, throughput_unit=throughput_unit,
        )

    def test_first_success_stores_perf_fields(self, tmp_bestiary):
        self._make_success(tmp_bestiary)
        e = tmp_bestiary.compiled["m"]
        assert e["best_compile_s"] == 5.0
        assert e["best_infer_s"] == 0.5
        assert e["best_throughput"] == 12.0
        assert e["throughput_unit"] == "tokens/sec"

    def test_tokens_per_sec_higher_is_better(self, tmp_bestiary):
        self._make_success(tmp_bestiary, throughput=10.0, throughput_unit="tokens/sec")
        self._make_success(tmp_bestiary, throughput=15.0, throughput_unit="tokens/sec")
        assert tmp_bestiary.compiled["m"]["best_throughput"] == 15.0

    def test_ms_per_sample_lower_is_better(self, tmp_bestiary):
        self._make_success(tmp_bestiary, throughput=100.0, throughput_unit="ms/sample")
        self._make_success(tmp_bestiary, throughput=80.0, throughput_unit="ms/sample")
        assert tmp_bestiary.compiled["m"]["best_throughput"] == 80.0

    def test_best_compile_s_tracks_minimum(self, tmp_bestiary):
        self._make_success(tmp_bestiary, compile_s=8.0)
        self._make_success(tmp_bestiary, compile_s=5.0)
        self._make_success(tmp_bestiary, compile_s=6.0)
        assert tmp_bestiary.compiled["m"]["best_compile_s"] == 5.0

    def test_zero_values_not_stored(self, tmp_bestiary):
        b = tmp_bestiary
        b.record_success(
            model_id="x", chip=0, run=1, time_s=10.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        assert "best_compile_s" not in b.compiled["x"]
        assert "best_infer_s" not in b.compiled["x"]
        assert "best_throughput" not in b.compiled["x"]

    def test_existing_tests_still_pass_without_new_args(self, tmp_bestiary):
        tmp_bestiary.record_success(
            model_id="legacy", chip=0, run=1, time_s=30.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        assert "legacy" in tmp_bestiary.compiled


class TestAppendPerfRecord:
    def test_appends_jsonl_line(self, tmp_bestiary):
        record = {"model_id": "gpt2/pytorch", "run": 1, "compile_s": 10.2,
                  "infer_s": 2.3, "throughput": 13.9, "throughput_unit": "tokens/sec"}
        tmp_bestiary.append_perf_record(record)
        perf_path = tmp_bestiary.path.parent / "perf_history.jsonl"
        assert perf_path.exists()
        lines = perf_path.read_text().strip().split("\n")
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["model_id"] == "gpt2/pytorch"
        assert loaded["compile_s"] == 10.2

    def test_appends_multiple_records(self, tmp_bestiary):
        for i in range(3):
            tmp_bestiary.append_perf_record({"run": i})
        perf_path = tmp_bestiary.path.parent / "perf_history.jsonl"
        lines = perf_path.read_text().strip().split("\n")
        assert len(lines) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 -m pytest tests/lib/test_bestiary.py::TestBestiaryPerfFields tests/lib/test_bestiary.py::TestAppendPerfRecord -v
```

Expected: `FAILED` — `record_success()` does not accept `compile_s` etc., and `append_perf_record` does not exist.

- [ ] **Step 3: Add compile_s/infer_s/throughput/throughput_unit params to record_success()**

In `lib/expedition/bestiary.py`, update the `record_success` signature (keep all existing params, add four keyword-only at the end with defaults):

```python
def record_success(
    self,
    model_id: str,
    chip: int,
    run: int,
    time_s: float,
    task: str,
    source: str,
    rarity: str,
    hf_downloads: int | None,
    hf_created_at: str | None,
    artifact: str,
    backend: str = "forge",
    first_voice: str = "",
    compile_s: float = 0.0,
    infer_s: float = 0.0,
    throughput: float = 0.0,
    throughput_unit: str = "",
) -> None:
```

After the existing `entry["successes"] += 1` and `best_time_s` update block, add:

```python
        # Update split timing and throughput rolling bests.
        if compile_s > 0.0:
            if "best_compile_s" not in entry or compile_s < entry["best_compile_s"]:
                entry["best_compile_s"] = compile_s
        if infer_s > 0.0:
            if "best_infer_s" not in entry or infer_s < entry["best_infer_s"]:
                entry["best_infer_s"] = infer_s
        if throughput > 0.0:
            entry["throughput_unit"] = throughput_unit
            if "best_throughput" not in entry:
                entry["best_throughput"] = throughput
            elif throughput_unit == "tokens/sec":
                if throughput > entry["best_throughput"]:
                    entry["best_throughput"] = throughput
            else:  # ms/sample — lower is better
                if throughput < entry["best_throughput"]:
                    entry["best_throughput"] = throughput
```

- [ ] **Step 4: Add append_perf_record method**

Add after `save_artifact` in `lib/expedition/bestiary.py`:

```python
    def append_perf_record(self, record: dict) -> None:
        """Append one performance record line to the sibling perf_history.jsonl file.

        The file is stored alongside bestiary.json as data/perf_history.jsonl.
        Each line is a self-contained JSON object — one per model per run.
        """
        perf_path = self.path.parent / "perf_history.jsonl"
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        with perf_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/lib/test_bestiary.py -v
```

Expected: all pass including the new classes.

- [ ] **Step 6: Commit**

```bash
git add lib/expedition/bestiary.py tests/lib/test_bestiary.py
git commit -m "feat: bestiary perf fields — compile_s, infer_s, throughput + append_perf_record"
```

---

## Task 2: Forge worker — split infer_time + _compute_throughput

**Files:**
- Modify: `lib/expedition/expedition_worker.py`
- Create: `tests/lib/test_perf_worker_helpers.py`

- [ ] **Step 1: Write failing tests for _compute_throughput**

Create `tests/lib/test_perf_worker_helpers.py`:

```python
import sys
import types
import torch
import pytest

# Stub out forge so the worker module can be imported without forge installed.
_forge_stub = types.ModuleType("forge")
sys.modules.setdefault("forge", _forge_stub)

from lib.expedition.expedition_worker import _compute_throughput


class TestComputeThroughput:
    def _tensor(self, shape):
        return torch.zeros(*shape)

    def test_text_generation_tokens_per_sec(self):
        output = self._tensor((1, 64, 50257))  # batch=1, seq=64, vocab
        tput, unit = _compute_throughput("text-generation", output, infer_s=2.0)
        assert unit == "tokens/sec"
        assert tput == pytest.approx(32.0)  # 64 tokens / 2s

    def test_nlp_causal_lm_tokens_per_sec(self):
        output = self._tensor((1, 32, 32000))
        tput, unit = _compute_throughput("nlp_causal_lm", output, infer_s=1.0)
        assert unit == "tokens/sec"
        assert tput == pytest.approx(32.0)

    def test_fill_mask_tokens_per_sec(self):
        output = self._tensor((1, 32, 30000))
        tput, unit = _compute_throughput("fill-mask", output, infer_s=0.5)
        assert unit == "tokens/sec"
        assert tput == pytest.approx(64.0)

    def test_image_classification_ms_per_sample(self):
        output = self._tensor((1, 1000))
        tput, unit = _compute_throughput("image-classification", output, infer_s=0.1)
        assert unit == "ms/sample"
        assert tput == pytest.approx(100.0)

    def test_cv_image_cls_ms_per_sample(self):
        output = self._tensor((1, 1000))
        tput, unit = _compute_throughput("cv_image_cls", output, infer_s=0.05)
        assert unit == "ms/sample"
        assert tput == pytest.approx(50.0)

    def test_zero_infer_s_returns_empty(self):
        output = self._tensor((1, 32, 50257))
        tput, unit = _compute_throughput("text-generation", output, infer_s=0.0)
        assert tput == 0.0
        assert unit == ""

    def test_none_output_returns_empty(self):
        tput, unit = _compute_throughput("text-generation", None, infer_s=1.0)
        assert tput == 0.0
        assert unit == ""

    def test_nlp_embed_gen_uses_ms_per_sample(self):
        output = self._tensor((1, 32, 1024))
        tput, unit = _compute_throughput("nlp_embed_gen", output, infer_s=0.2)
        assert unit == "ms/sample"

    def test_nlp_qa_uses_ms_per_sample(self):
        output = self._tensor((1, 32))
        tput, unit = _compute_throughput("nlp_qa", output, infer_s=0.1)
        assert unit == "ms/sample"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/lib/test_perf_worker_helpers.py -v
```

Expected: `ImportError` — `_compute_throughput` does not exist yet.

- [ ] **Step 3: Add _TOKEN_TASKS and _compute_throughput to expedition_worker.py**

Add after the `_ERROR_RULES` list near the top of `lib/expedition/expedition_worker.py` (around line 120, after `_CSV_FIELDNAMES`):

```python
# Tasks whose output tensor's seq dimension is meaningful as token count.
_TOKEN_TASKS: frozenset[str] = frozenset({
    "text-generation", "nlp_causal_lm", "nlp_masked_lm",
    "fill-mask", "nlp_text_cls", "nlp_token_cls",
})


def _compute_throughput(task: str, output: Any, infer_s: float) -> tuple[float, str]:
    """Compute throughput from a decoded inference output tensor.

    For token-producing tasks, returns (tokens_per_sec, "tokens/sec") using
    output.shape[1] as the sequence length.  For all other tasks (CV, embeddings,
    QA, audio) returns (ms_per_sample, "ms/sample").  Returns (0.0, "") when
    infer_s is zero or the output shape cannot be read.
    """
    if infer_s <= 0.0 or output is None:
        return 0.0, ""
    try:
        if task in _TOKEN_TASKS:
            seq_len = output.shape[1]
            return round(seq_len / infer_s, 3), "tokens/sec"
    except (AttributeError, IndexError):
        pass
    return round(infer_s * 1000.0, 3), "ms/sample"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/lib/test_perf_worker_helpers.py -v
```

Expected: all pass.

- [ ] **Step 5: Extend _compile_model return tuple to include infer_time and sample_inputs**

In `lib/expedition/expedition_worker.py`, update `_compile_model`:

Change the docstring return description (around line 328):

```python
    Returns a 7-tuple (success, output, compile_time, infer_time, error_str, compiled, sample_inputs):
      - success:       True if both compile and inference completed without error.
      - output:        The raw inference output tensor/list (None on failure).
      - compile_time:  Seconds spent in forge.compile() (0.0 on failure before compile).
      - infer_time:    Seconds spent on the first inference pass (0.0 on failure or before infer).
      - error_str:     Empty string on success; "TIMEOUT" or "ExcType: msg" on failure.
      - compiled:      The forge-compiled module (None on failure).
      - sample_inputs: Normalised input tensors used for compilation ([] on failure).
```

Change the function signature annotation:

```python
def _compile_model(model_loader, chip_id: int, timeout: int = 120) -> tuple[bool, Any, float, float, str, Any, list]:
```

Add `infer_time` measurement (around line 478-488). Replace the inference block:

```python
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
```

Update the unwrap + success return (around line 491-494):

```python
        # Unwrap list outputs (forge sometimes returns [tensor]).
        if isinstance(output, list):
            output = output[0] if output else None

        return True, output, compile_time, infer_time, "", compiled, list(sample_inputs)
```

Update all failure returns to include the two new slots. Find and replace each `return False, None, compile_time, f"...`, None` with `return False, None, compile_time, 0.0, "...", None, []`.

The four failure return sites are:
1. `return False, None, compile_time, f"{type(e).__name__}: {str(e)[:300]}", None` → `return False, None, compile_time, 0.0, f"{type(e).__name__}: {str(e)[:300]}", None, []`
2. `return False, None, compile_time, "TIMEOUT", None` (inside inference block) → already updated above
3. `return False, None, 0.0, "TIMEOUT", None` (outer try/except) → `return False, None, 0.0, 0.0, "TIMEOUT", None, []`
4. `return False, None, 0.0, f"{type(e).__name__}: {str(e)[:300]}", None` → `return False, None, 0.0, 0.0, f"{type(e).__name__}: {str(e)[:300]}", None, []`

- [ ] **Step 6: Update the three call sites in run_worker**

In `run_worker` (around lines 862 and 866), update both unpack lines:

```python
        success, output, compile_time, infer_time, error_str, compiled_module, sample_inputs = _compile_model(loader, chip_id)
        # Auto-install missing packages and retry once
        if not success and "No module named" in error_str:
            if _try_install_missing(error_str):
                success, output, compile_time, infer_time, error_str, compiled_module, sample_inputs = _compile_model(loader, chip_id)
```

- [ ] **Step 7: Run existing tests to verify nothing broke**

```bash
python3 -m pytest tests/lib/ -v --ignore=tests/lib/test_perf_worker_helpers.py -k "not forge"
```

Expected: all existing tests pass (bestiary, decoder, scorer, hud, router, dispatch).

- [ ] **Step 8: Commit**

```bash
git add lib/expedition/expedition_worker.py tests/lib/test_perf_worker_helpers.py
git commit -m "feat: split infer_time from _compile_model return; add _compute_throughput helper"
```

---

## Task 3: Forge worker — _run_bench_passes + _run_shape_sweep

**Files:**
- Modify: `lib/expedition/expedition_worker.py`
- Test: `tests/lib/test_perf_worker_helpers.py`

- [ ] **Step 1: Write failing tests for _run_bench_passes**

Add to `tests/lib/test_perf_worker_helpers.py`:

```python
from lib.expedition.expedition_worker import _run_bench_passes
import time


class TestRunBenchPasses:
    def _make_compiled(self, sleep_s=0.01):
        """Return a callable that fakes a compiled module with known latency."""
        import torch

        def fake_compiled(*args):
            time.sleep(sleep_s)
            return torch.zeros(1, 32, 50257)

        return fake_compiled

    def test_returns_expected_keys(self):
        compiled = self._make_compiled(sleep_s=0.01)
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=3, task="text-generation")
        assert set(result.keys()) == {
            "bench_passes", "infer_p50_s", "infer_p95_s",
            "throughput_p50", "throughput_p95",
        }

    def test_bench_passes_count_matches(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=5, task="text-generation")
        assert result["bench_passes"] == 5

    def test_p50_le_p95(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=5, task="text-generation")
        assert result["infer_p50_s"] <= result["infer_p95_s"]

    def test_throughput_unit_tokens_per_sec(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=3, task="text-generation")
        assert result["throughput_p50"] > 0

    def test_zero_passes_returns_empty(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=0, task="text-generation")
        assert result == {}

    def test_crashing_compiled_returns_empty(self):
        def crashing(*args):
            raise RuntimeError("simulated crash")

        import torch
        inputs = [torch.zeros(1, 32)]
        result = _run_bench_passes(crashing, inputs, n_passes=5, task="text-generation")
        assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/lib/test_perf_worker_helpers.py::TestRunBenchPasses -v
```

Expected: `ImportError` — `_run_bench_passes` does not exist yet.

- [ ] **Step 3: Add _percentile helper and _run_bench_passes to expedition_worker.py**

Add immediately after `_compute_throughput` in `lib/expedition/expedition_worker.py`:

```python
def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) from a sorted list via linear interpolation."""
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
    Stops silently on error — returns partial results when some passes succeed.

    Args:
        compiled_module: forge-compiled module from forge.compile().
        sample_inputs:   Normalised input tensor list (as returned by _compile_model).
        n_passes:        Number of timed passes.  0 returns {}.
        task:            HuggingFace task string — controls throughput unit.

    Returns a dict with bench_passes, infer_p50_s, infer_p95_s,
    throughput_p50, throughput_p95.  Empty dict on failure or n_passes <= 0.
    """
    if n_passes <= 0:
        return {}
    last_output = None
    # Warm-up: 2 passes, not timed.
    for _ in range(2):
        try:
            compiled_module(*sample_inputs)
        except Exception:
            return {}
    # Timed passes.
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
```

- [ ] **Step 4: Add _run_shape_sweep to expedition_worker.py**

Add immediately after `_run_bench_passes`:

```python
def _run_shape_sweep(
    compiled_module: Any,
    loader: Any,
    task: str,
    n_passes: int,
) -> list[dict]:
    """Run bench passes at alternative input shapes.

    For token tasks: sweeps seq=[128, 512] at batch=1.
    For vision/other: sweeps image size 384 (if default was 224).
    Shape-level failures are silently skipped.

    Returns a list of dicts, each with the shape key(s) plus infer_s and throughput.
    """
    import torch
    itype = getattr(loader, "_input_type", "image")
    if task in _TOKEN_TASKS:
        sweep = [{"seq": 128}, {"seq": 512}]
        def make_inputs(spec):
            return [torch.randint(0, 1000, (1, spec["seq"]))]
    else:
        sweep = [{"img_size": 384}]
        def make_inputs(spec):
            return [torch.randn(1, 3, spec["img_size"], spec["img_size"])]

    results: list[dict] = []
    for spec in sweep:
        try:
            inputs = _normalise_inputs(make_inputs(spec))
            times: list[float] = []
            last_out = None
            for _ in range(2):  # warm-up
                compiled_module(*inputs)
            for _ in range(n_passes):
                t0 = time.time()
                last_out = compiled_module(*inputs)
                times.append(time.time() - t0)
            if times:
                times.sort()
                p50 = _percentile(times, 50)
                tput, _ = _compute_throughput(task, last_out, p50)
                results.append({**spec, "infer_s": round(p50, 4), "throughput": round(tput, 2)})
        except Exception:
            pass  # skip shapes that fail
    return results
```

- [ ] **Step 5: Run all perf helper tests**

```bash
python3 -m pytest tests/lib/test_perf_worker_helpers.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/expedition/expedition_worker.py tests/lib/test_perf_worker_helpers.py
git commit -m "feat: add _run_bench_passes and _run_shape_sweep helpers to forge worker"
```

---

## Task 4: Forge worker — wire run_worker, update _print_success, CSV fields, CLI args

**Files:**
- Modify: `lib/expedition/expedition_worker.py`

- [ ] **Step 1: Update _CSV_FIELDNAMES to include infer_time**

In `lib/expedition/expedition_worker.py`, change line ~120:

```python
_CSV_FIELDNAMES = [
    "model", "status", "pts", "compile_time", "infer_time",
    "artifact", "first_ever", "first_voice", "error",
]
```

- [ ] **Step 2: Update _print_success signature and body**

Replace the existing `_print_success` function (around line 243):

```python
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
    streak_str = f"  {TEAL}🔥×{streak}{RESET}" if streak >= 3 else ""
    first_str  = f"  {GOLD}★ FIRST{RESET}" if is_first_ever else ""
    if bench_p50 > 0.0 and throughput_unit:
        tput_str = f"  ~{throughput:.1f} {throughput_unit} (p50)"
    elif throughput > 0.0 and throughput_unit:
        tput_str = f"  {throughput:.1f} {throughput_unit}"
    else:
        tput_str = ""
    print(f"    compile: {compile_time:.1f}s  infer: {infer_time:.2f}s"
          f"{tput_str}  pts: {GOLD}{score_pts:+d}{RESET}"
          f"{streak_str}{first_str}")
    print(f"    {DIM}{artifact[:80]}{RESET}")
```

- [ ] **Step 3: Update the _print_success call site in run_worker**

Around line 895 in `run_worker`, the existing call is:
```python
_print_success(item.model_id, compile_time, elapsed, artifact,
               score.pts, is_first_ever, hud.state.streak)
```

Replace with (after computing throughput below):
```python
_print_success(
    item.model_id, compile_time, infer_time, artifact,
    score.pts, is_first_ever, hud.state.streak,
    throughput=throughput, throughput_unit=throughput_unit,
    bench_p50=bench_stats.get("infer_p50_s", 0.0),
)
```

- [ ] **Step 4: Wire bench passes, perf record, and updated record_success call in run_worker**

In the success block of `run_worker` (after `_attempt_first_voice` and before `bestiary.save_artifact`), add throughput computation, optional bench loop, and perf record write. Replace the existing `bestiary.record_success(...)` call with the updated one.

The full success block from after `is_first_voice = bool(first_voice_text)` through `results.append(...)` should read:

```python
            # Compute throughput from the standard inference pass.
            throughput, throughput_unit = _compute_throughput(
                item.task, output, infer_time
            )

            score = compute_score(success=True, is_first_ever=is_first_ever,
                                  rarity=rarity, newness=newness,
                                  streak=hud.state.streak,
                                  mesh_chips=item.mesh_chips,
                                  is_first_voice=is_first_voice)
            hud.record_success(item.model_id, score)

            # Optional bench passes (--bench-passes N).
            bench_stats: dict = {}
            if bench_passes > 0 and compiled_module is not None and sample_inputs:
                bench_stats = _run_bench_passes(
                    compiled_module, sample_inputs, bench_passes, item.task
                )

            _print_success(
                item.model_id, compile_time, infer_time, artifact,
                score.pts, is_first_ever, hud.state.streak,
                throughput=throughput, throughput_unit=throughput_unit,
                bench_p50=bench_stats.get("infer_p50_s", 0.0),
            )

            if is_first_voice and first_voice_sample:
                print(f"    {GOLD}🗣 First Voice{RESET}  "
                      f"{DIM}[{first_voice_sample['description']}]{RESET}")
                print(f"    {PINK}{first_voice_text}{RESET}")

            compiled_at = datetime.datetime.now().isoformat()

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
                    pass

            bestiary.save_artifact(
                model_id=item.model_id,
                task=item.task,
                compiled_at=compiled_at,
                chip=chip_id,
                run=run_number,
                artifact_text=first_voice_text if is_first_voice else artifact,
            )

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
            perf_record.update(bench_stats)
            # Optional shape sweep (--bench-shapes requires --bench-passes > 0).
            if bench_shapes and bench_passes > 0 and compiled_module is not None:
                shapes = _run_shape_sweep(
                    compiled_module, loader, item.task, bench_passes
                )
                if shapes:
                    perf_record["shapes"] = shapes
            bestiary.append_perf_record(perf_record)

            bestiary.add_chip_points(
                chip=chip_id,
                pts=score.pts,
                first_ever=is_first_ever,
                streak=hud.state.streak,
            )
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
```

- [ ] **Step 5: Add bench_passes and bench_shapes params to run_worker signature**

Update `run_worker` signature:

```python
def run_worker(chip_id: int, run_number: int, bestiary_path: str,
               queue_path: str | None, results_path: str,
               model_json_path: str | None = None,
               bench_passes: int = 0,
               bench_shapes: bool = False) -> None:
```

Update the docstring args section to include:
```
        bench_passes:    Number of timed inference passes after each successful compile.
                         0 (default) disables benchmarking.
        bench_shapes:    If True and bench_passes > 0, also sweep alternative input shapes.
```

- [ ] **Step 6: Add --bench-passes and --bench-shapes to the __main__ argument parser**

In the `if __name__ == "__main__":` block (around line 1017), add after `--results`:

```python
    parser.add_argument("--bench-passes", type=int, default=0, metavar="N",
                        help="Run 2 warm-up + N timed inference passes after each "
                             "successful compile. Default 0 (disabled).")
    parser.add_argument("--bench-shapes", action="store_true",
                        help="Sweep alternative input shapes after bench passes. "
                             "Requires --bench-passes > 0.")
```

Update the `run_worker(...)` call at the bottom of `__main__`:

```python
    run_worker(
        chip_id=args.chip,
        run_number=args.run,
        bestiary_path=args.bestiary,
        queue_path=args.queue,
        model_json_path=args.model_json,
        results_path=args.results,
        bench_passes=args.bench_passes,
        bench_shapes=args.bench_shapes,
    )
```

- [ ] **Step 7: Verify --help shows new flags**

```bash
python3 lib/expedition/expedition_worker.py --help
```

Expected: `--bench-passes N` and `--bench-shapes` appear in the help output.

- [ ] **Step 8: Run all tests**

```bash
python3 -m pytest tests/lib/ -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "feat: wire bench passes, throughput, and perf_history into forge run_worker"
```

---

## Task 5: XLA worker — infer_time split + throughput + bench passes + CLI args

**Files:**
- Modify: `lib/expedition/expedition_worker_xla.py`

The XLA worker's JIT compiles lazily on the first call, so `compile_time` includes
compile + first infer together. Strategy: after JIT completes, run one additional
timed inference pass to get a pure `infer_s`. The `compile_s` stored is
`compile_time - infer_s` (approximate; documented in the JSONL `"backend": "xla"`).

- [ ] **Step 1: Add _TOKEN_TASKS and _compute_throughput to expedition_worker_xla.py**

Add after the `_CSV_FIELDNAMES` constant near the top of `lib/expedition/expedition_worker_xla.py` (around line 123):

```python
_TOKEN_TASKS: frozenset[str] = frozenset({
    "text-generation", "nlp_causal_lm", "nlp_masked_lm",
    "fill-mask", "nlp_text_cls", "nlp_token_cls",
})


def _compute_throughput_xla(task: str, output: Any, infer_s: float) -> tuple[float, str]:
    """Compute throughput for XLA output (JAX arrays).

    Same logic as the forge worker but handles JAX array shapes.
    Returns (tokens_per_sec, "tokens/sec") for token tasks,
    (ms_per_sample, "ms/sample") otherwise.  Returns (0.0, "") on failure.
    """
    if infer_s <= 0.0 or output is None:
        return 0.0, ""
    try:
        if task in _TOKEN_TASKS:
            seq_len = output.shape[1]
            return round(seq_len / infer_s, 3), "tokens/sec"
    except (AttributeError, IndexError):
        pass
    return round(infer_s * 1000.0, 3), "ms/sample"
```

- [ ] **Step 2: Add a second inference pass to _compile_model_xla for infer_s measurement**

In `lib/expedition/expedition_worker_xla.py`, update `_compile_model_xla` to:
1. Measure the first JIT call (unchanged, stays as `compile_time`)
2. Run a second timed pass for pure `infer_s`
3. Return a 6-tuple: `(success, output, compile_time, infer_s, error_str, compiled_bundle)`

Update the docstring return:
```python
    Returns 6-tuple (success, output, compile_time, infer_s, error_str, compiled_bundle):
      - compile_time: Seconds for JIT compilation + first inference (inseparable in JAX).
      - infer_s:      Seconds for a second inference pass (pure inference, JIT already done).
                      0.0 if the second pass fails or times out.
```

For the **multi-chip path** (around line 375-386), after the existing success block:

```python
            compile_time = time.time() - compile_start
            _print_progress_step(3, 3, f"Output shape: {output.shape}  ({compile_time:.1f}s — {n} chips)")

            # Second pass for pure infer_s measurement (JIT already done).
            infer_s = 0.0
            try:
                infer_start = time.time()
                _ = compiled_fn(sharded_params, sharded_inputs)
                _.block_until_ready()
                infer_s = time.time() - infer_start
            except Exception:
                pass

            return True, output, compile_time, infer_s, "", (compiled_fn, sharded_params, all_devices[0])
```

For the **single-chip path** (around line 396-412), after the existing success block:

```python
            compile_time = time.time() - compile_start
            _print_progress_step(3, 3, f"Output shape: {output.shape}  ({compile_time:.1f}s)")

            # Second pass for pure infer_s measurement (JIT already done).
            infer_s = 0.0
            try:
                with jax.default_device(device):
                    infer_start = time.time()
                    _ = compiled_fn(flax_params, dummy_inputs)
                    _.block_until_ready()
                    infer_s = time.time() - infer_start
            except Exception:
                pass

            return True, output, compile_time, infer_s, "", (compiled_fn, flax_params, device)
```

Update all failure returns to the 6-tuple form:
- `return False, None, time.time() - compile_start, 0.0, "TIMEOUT", None`
- `return False, None, 0.0, 0.0, "TIMEOUT", None`
- `return False, None, 0.0, 0.0, f"{type(e).__name__}: {str(e)[:300]}", None`

- [ ] **Step 3: Update _print_success in the XLA worker**

Replace the existing `_print_success` in `lib/expedition/expedition_worker_xla.py` (around line 187):

```python
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
    streak_str = f"  {TEAL}🔥×{streak}{RESET}" if streak >= 3 else ""
    first_str  = f"  {GOLD}★ FIRST{RESET}" if is_first_ever else ""
    if bench_p50 > 0.0 and throughput_unit:
        tput_str = f"  ~{throughput:.1f} {throughput_unit} (p50)"
    elif throughput > 0.0 and throughput_unit:
        tput_str = f"  {throughput:.1f} {throughput_unit}"
    else:
        tput_str = ""
    print(f"    compile: {compile_time:.1f}s  infer: {infer_time:.2f}s"
          f"{tput_str}  pts: {GOLD}{score_pts:+d}{RESET}"
          f"{streak_str}{first_str}")
    print(f"    {DIM}{artifact[:80]}{RESET}")
```

- [ ] **Step 4: Add _run_bench_passes_xla helper**

Add after `_compute_throughput_xla` in `lib/expedition/expedition_worker_xla.py`:

```python
def _percentile_xla(sorted_data: list[float], p: float) -> float:
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

    compiled_bundle is (compiled_fn, params, device) from _compile_model_xla.
    Warm-up: 2 passes. Then n_passes timed.  Stops on error.
    Returns dict with bench_passes, infer_p50_s, infer_p95_s,
    throughput_p50, throughput_p95.  Empty dict on failure or n_passes <= 0.
    """
    if n_passes <= 0 or compiled_bundle is None:
        return {}
    try:
        compiled_fn, params, device = compiled_bundle
        # Reconstruct dummy inputs — XLA bench uses the same shape as the JIT run.
        import jax, jax.numpy as jnp
        dummy = jnp.ones((1, 32), dtype=jnp.int32)
        with jax.default_device(device):
            for _ in range(2):  # warm-up
                out = compiled_fn(params, dummy)
                out.block_until_ready()
        times: list[float] = []
        last_out = None
        with jax.default_device(device):
            for _ in range(n_passes):
                try:
                    t0 = time.time()
                    last_out = compiled_fn(params, dummy)
                    last_out.block_until_ready()
                    times.append(time.time() - t0)
                except Exception:
                    break
        if not times:
            return {}
        times.sort()
        p50 = _percentile_xla(times, 50)
        p95 = _percentile_xla(times, 95)
        tput_p50, _ = _compute_throughput_xla(task, last_out, p50)
        tput_p95, _ = _compute_throughput_xla(task, last_out, p95)
        return {
            "bench_passes":   len(times),
            "infer_p50_s":    round(p50, 4),
            "infer_p95_s":    round(p95, 4),
            "throughput_p50": round(tput_p50, 2),
            "throughput_p95": round(tput_p95, 2),
        }
    except Exception:
        return {}
```

- [ ] **Step 5: Update _CSV_FIELDNAMES in expedition_worker_xla.py**

Change the `_CSV_FIELDNAMES` list (around line 123):

```python
_CSV_FIELDNAMES = [
    "model", "status", "pts", "compile_time", "infer_time",
    "artifact", "first_ever", "first_voice", "error", "backend",
]
```

- [ ] **Step 6: Update run_worker_xla call sites and success block**

Update both unpack lines in `run_worker_xla` (around lines 866 and 872):

```python
        success, output, compile_time, infer_time, error_str, compiled_bundle = _compile_model_xla(
            loader, device, chip_id, mesh_chips=item.mesh_chips
        )
        if not success and "No module named" in error_str:
            if _try_install_missing(error_str):
                success, output, compile_time, infer_time, error_str, compiled_bundle = _compile_model_xla(
                    loader, device, chip_id, mesh_chips=item.mesh_chips
                )
```

In the success block, add throughput, bench stats, and perf record (same pattern as forge worker Task 4 Step 4). After `is_first_voice = bool(first_voice_text)`:

```python
            # Compute throughput from the second inference pass (infer_time).
            throughput, throughput_unit = _compute_throughput_xla(
                item.task, output, infer_time
            )

            score = compute_score(success=True, is_first_ever=is_first_ever,
                                  rarity=rarity, newness=newness,
                                  streak=hud.state.streak,
                                  mesh_chips=item.mesh_chips,
                                  is_first_voice=is_first_voice)
            hud.record_success(item.model_id, score)

            bench_stats: dict = {}
            if bench_passes > 0 and compiled_bundle is not None:
                bench_stats = _run_bench_passes_xla(compiled_bundle, item.task, bench_passes)

            _print_success(
                item.model_id, compile_time, infer_time, artifact,
                score.pts, is_first_ever, hud.state.streak,
                throughput=throughput, throughput_unit=throughput_unit,
                bench_p50=bench_stats.get("infer_p50_s", 0.0),
            )
```

Replace `bestiary.record_success(...)` with:

```python
            bestiary.record_success(
                model_id=item.model_id, chip=chip_id, run=run_number,
                time_s=compile_time, task=item.task, source=item.source,
                rarity=rarity.value, hf_downloads=item.hf_downloads,
                hf_created_at=item.hf_created_at,
                artifact=first_voice_text if is_first_voice else artifact,
                backend=BACKEND_LABEL,
                compile_s=compile_time,
                infer_s=infer_time,
                throughput=throughput,
                throughput_unit=throughput_unit,
            )

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
```

Update `results.append(...)` to include `"infer_time": infer_time`.

Update `run_worker_xla` signature to add `bench_passes: int = 0, bench_shapes: bool = False`.

- [ ] **Step 7: Add --bench-passes and --bench-shapes to XLA worker __main__ parser**

In the `if __name__ == "__main__":` block of `expedition_worker_xla.py`, add after `--results`:

```python
    parser.add_argument("--bench-passes", type=int, default=0, metavar="N",
                        help="Run 2 warm-up + N timed inference passes after each "
                             "successful compile. Default 0 (disabled).")
    parser.add_argument("--bench-shapes", action="store_true",
                        help="Sweep alternative input shapes (forge-only; XLA ignores this flag).")
```

Update the `run_worker_xla(...)` call to pass `bench_passes=args.bench_passes, bench_shapes=args.bench_shapes`.

- [ ] **Step 8: Verify --help shows new flags**

```bash
python3 lib/expedition/expedition_worker_xla.py --help
```

Expected: `--bench-passes N` and `--bench-shapes` appear.

- [ ] **Step 9: Run all tests**

```bash
python3 -m pytest tests/lib/ -v
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add lib/expedition/expedition_worker_xla.py
git commit -m "feat: XLA worker — split infer_time, throughput, bench passes, CLI args"
```

---

## Task 6: run_state.py — add infer_time to ModelResult

**Files:**
- Modify: `lib/expedition/run_state.py`
- Create: `tests/lib/test_run_state.py`

- [ ] **Step 1: Write failing tests**

Create `tests/lib/test_run_state.py`:

```python
from lib.expedition.run_state import ModelResult


class TestModelResultInferTime:
    def _make_row(self, infer_time="0.42"):
        return {
            "model": "gpt2/pytorch",
            "status": "success",
            "pts": "250",
            "compile_time": "10.5",
            "infer_time": infer_time,
            "artifact": "shape=(1,32,50257)",
            "first_ever": "True",
            "first_voice": "False",
            "error": "",
        }

    def test_from_csv_row_parses_infer_time(self):
        row = self._make_row(infer_time="0.42")
        result = ModelResult.from_csv_row(row, chip_id=0, rarity="common", streak=1)
        assert result.infer_time == 0.42

    def test_from_csv_row_defaults_infer_time_to_zero(self):
        row = self._make_row()
        del row["infer_time"]
        result = ModelResult.from_csv_row(row, chip_id=0, rarity="common", streak=1)
        assert result.infer_time == 0.0

    def test_from_csv_row_handles_empty_infer_time(self):
        row = self._make_row(infer_time="")
        result = ModelResult.from_csv_row(row, chip_id=0, rarity="common", streak=1)
        assert result.infer_time == 0.0

    def test_infer_time_field_exists_on_model_result(self):
        result = ModelResult(
            chip_id=0, model_id="x", status="success", pts=100,
            compile_time=5.0, infer_time=0.5, artifact="", first_ever=False,
            first_voice=False, error="", rarity="common", streak=1,
        )
        assert result.infer_time == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/lib/test_run_state.py -v
```

Expected: `TypeError` — `ModelResult` does not have `infer_time`.

- [ ] **Step 3: Add infer_time to ModelResult dataclass**

In `lib/expedition/run_state.py`, add `infer_time: float` to the `ModelResult` dataclass after `compile_time`:

```python
@dataclass
class ModelResult:
    """One compile/infer attempt for a single model on a single chip.

    Attributes:
        chip_id:      Zero-based chip index that ran this model.
        model_id:     Model identifier (e.g. "gpt2/pytorch").
        status:       "success" or "failed".
        pts:          Points awarded (positive for success, negative for failure).
        compile_time: Wall-clock forge.compile() time in seconds (0.0 on failure).
        infer_time:   Wall-clock first inference pass time in seconds (0.0 on failure).
        artifact:     Human-readable inference output summary, or "".
        first_ever:   True if this is the first successful compile of this model.
        first_voice:  True if the model produced decoded text output this run.
        error:        Error string on failure (may be multi-line), or "".
        rarity:       Rarity tier: "legendary" | "rare" | "uncommon" | "common".
        streak:       Consecutive-success streak count at the time of this result.
        is_sq:        True for side quest (bonus) models; False for main queue.
    """

    chip_id: int
    model_id: str
    status: str
    pts: int
    compile_time: float
    infer_time: float
    artifact: str
    first_ever: bool
    first_voice: bool
    error: str
    rarity: str
    streak: int
    is_sq: bool = False
```

- [ ] **Step 4: Update from_csv_row**

In `from_csv_row`, add `infer_time` after `compile_time`:

```python
        return cls(
            chip_id=chip_id,
            model_id=row.get("model", ""),
            status=row.get("status", "failed"),
            pts=int(row.get("pts") or 0),
            compile_time=float(row.get("compile_time") or 0.0),
            infer_time=float(row.get("infer_time") or 0.0),
            artifact=row.get("artifact", ""),
            first_ever=row.get("first_ever") == "True",
            first_voice=row.get("first_voice") == "True",
            error=row.get("error", ""),
            rarity=rarity,
            streak=streak,
            is_sq=is_sq,
        )
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest tests/lib/test_run_state.py tests/lib/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/expedition/run_state.py tests/lib/test_run_state.py
git commit -m "feat: add infer_time field to ModelResult and from_csv_row"
```

---

## Task 7: expedition_tui.py — forward --bench-passes and --bench-shapes

**Files:**
- Modify: `expedition_tui.py`

The TUI spawns worker subprocesses at two sites. Both need to forward the new flags when the top-level `expedition.py run` received them. The flags are passed via environment variables (simplest approach that avoids restructuring the subprocess call signature).

- [ ] **Step 1: Add --bench-passes and --bench-shapes to expedition.py run subcommand**

In `expedition.py`, add to the `run_p` argument parser (after `--auto-quit`, around line 1741):

```python
    run_p.add_argument("--bench-passes", type=int, default=0, metavar="N",
                       help="Inline bench: run 2 warm-up + N timed inference passes "
                            "after each successful compile. Default 0 (disabled).")
    run_p.add_argument("--bench-shapes", action="store_true",
                       help="Also sweep input shapes after bench passes. "
                            "Requires --bench-passes > 0.")
```

- [ ] **Step 2: Thread bench flags into the env passed to the TUI**

In `expedition.py`, find where `EXPEDITION_RUN` env is built (around line 1882):

```python
    env = {**os.environ, "EXPEDITION_RUN": str(run_number),
           "EXPEDITION_NUM_CHIPS": str(num_chips)}
```

Add the bench flags:

```python
    env = {**os.environ,
           "EXPEDITION_RUN":         str(run_number),
           "EXPEDITION_NUM_CHIPS":   str(num_chips),
           "EXPEDITION_BENCH_PASSES": str(getattr(args, "bench_passes", 0)),
           "EXPEDITION_BENCH_SHAPES": "1" if getattr(args, "bench_shapes", False) else "0",
           }
```

- [ ] **Step 3: Read bench env vars in expedition_tui.py and forward to both worker dispatch sites**

In `expedition_tui.py`, find where `python_exe` and `worker_path` are set up before the subprocess dispatch (around line 1204). Add env var reads near that setup point (search for where `EXPEDITION_RUN` is read, typically near the top of the `RunScreen` or dispatch method):

```python
_BENCH_PASSES = int(os.environ.get("EXPEDITION_BENCH_PASSES", "0"))
_BENCH_SHAPES = os.environ.get("EXPEDITION_BENCH_SHAPES", "0") == "1"
```

**First dispatch site** (subprocess.run, around line 1264 — the RALLY/mesh path):

```python
            cmd = [
                python_exe, worker_path,
                "--chip",       str(chip_id),
                "--run",        str(self.run_number),
                "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
                "--model-json", model_json_path,
                "--results",    results_path,
            ]
            if _BENCH_PASSES > 0:
                cmd += ["--bench-passes", str(_BENCH_PASSES)]
            if _BENCH_SHAPES:
                cmd.append("--bench-shapes")
```

**Second dispatch site** (asyncio.create_subprocess_exec, around line 1304):

```python
        args_extra = []
        if _BENCH_PASSES > 0:
            args_extra += ["--bench-passes", str(_BENCH_PASSES)]
        if _BENCH_SHAPES:
            args_extra.append("--bench-shapes")

        proc = await asyncio.create_subprocess_exec(
            python_exe,
            worker_path,
            "--chip",       str(chip_id),
            "--run",        str(self.run_number),
            "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
            "--model-json", model_json_path,
            "--results",    results_path,
            *args_extra,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
```

Also find and update the side quest dispatch site (around line 1411) with the same `*args_extra` pattern.

- [ ] **Step 4: Verify --help shows new flags**

```bash
python3 expedition.py run --help
```

Expected: `--bench-passes N` and `--bench-shapes` appear.

- [ ] **Step 5: Run all tests**

```bash
python3 -m pytest tests/lib/ -v
```

Expected: all pass.

- [ ] **Step 6: Verify dispatch flag check in intelligent dispatch test**

```bash
python3 -m pytest tests/lib/test_intelligent_dispatch.py -v -k "accepts"
```

Expected: `test_forge_worker_accepts_model_json_flag` and `test_xla_worker_accepts_model_json_flag` pass. (The new flags don't affect these tests.)

- [ ] **Step 7: Commit**

```bash
git add expedition.py expedition_tui.py
git commit -m "feat: thread --bench-passes and --bench-shapes through expedition.py and TUI dispatch"
```

---

## Self-Review

**Spec coverage:**
- ✅ `best_compile_s`, `best_infer_s`, `best_throughput`, `throughput_unit` in bestiary (Task 1)
- ✅ `data/perf_history.jsonl` append (Task 1)
- ✅ `compile_s`/`infer_s` separation (Tasks 2, 5)
- ✅ Throughput: tokens/sec for LLMs, ms/sample for others (Tasks 2, 5)
- ✅ `--bench-passes N` with warm-up + p50/p95 (Tasks 3, 4)
- ✅ `--bench-shapes` with LLM seq sweep and vision size sweep (Tasks 3, 4)
- ✅ TUI success line shows split times + throughput (Task 4)
- ✅ `total:` removed from TUI display (Task 4)
- ✅ XLA worker parallel changes (Task 5)
- ✅ `ModelResult.infer_time` for TUI (Task 6)
- ✅ CLI flags forwarded end-to-end (Task 7)

**Placeholder scan:** No TBDs. All steps have actual code.

**Type consistency:**
- `_compute_throughput` defined in Task 2, used in Tasks 3, 4 ✅
- `_run_bench_passes` defined in Task 3, used in Task 4 ✅
- `append_perf_record(record)` defined in Task 1, called in Tasks 4, 5 ✅
- `record_success(..., compile_s=, infer_s=, throughput=, throughput_unit=)` defined in Task 1, called in Tasks 4, 5 ✅
- `ModelResult(infer_time=)` defined in Task 6; CSV writes `infer_time` in Tasks 4, 5 ✅
