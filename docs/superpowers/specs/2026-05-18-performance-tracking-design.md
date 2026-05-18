# Performance Tracking & Benchmarking — Design Spec
**Date:** 2026-05-18  
**Status:** Approved

## Problem

The bestiary today stores one `best_time_s` per model — a single wall-clock number that conflates forge compile time with inference time. There is no inference throughput (tokens/sec, ms/sample), no long-term history of how a model's performance changes across runs, and no way to stress-test compiled models under varied inputs. This makes it impossible to answer: "Is this model getting faster? How many tokens per second does it produce? What happens at batch size 4?"

## Approach: Split bestiary summary from a separate perf log

The bestiary remains the source of truth for "did it compile." It gains four new summary fields per entry (rolling best values). A new append-only `data/perf_history.jsonl` captures the full detail of every performance measurement — one line per model per run. Optional CLI flags activate inline benchmarking (multiple inference passes, input shape sweeps) without affecting normal expedition runs.

## Section 1 — Data Schema

### Bestiary changes

Four new fields added to each `compiled` entry alongside the existing `best_time_s` (kept for backward compatibility):

```
best_compile_s   float  — fastest forge.compile() time seen across all runs
best_infer_s     float  — fastest single inference pass time (post-compile)
best_throughput  float  — best throughput: tokens/sec for LLMs, ms/sample for all others
throughput_unit  str    — "tokens/sec" | "ms/sample"
```

All four are updated with rolling-minimum logic inside `Bestiary.record_success()`, the same pattern already used for `best_time_s`.

### New file: `data/perf_history.jsonl`

Append-only. One JSON line per model per run. Never rewritten — only appended to. Provides the raw timeseries for long-term analysis.

Base fields (present on every line):

```json
{
  "model_id": "gpt2/pytorch",
  "run": 60,
  "timestamp": "2026-05-18T20:17:30+00:00",
  "backend": "forge",
  "chip": 1,
  "compile_s": 10.2,
  "infer_s": 2.3,
  "throughput": 13.9,
  "throughput_unit": "tokens/sec"
}
```

Optional bench fields (present only when `--bench-passes N` was passed):

```json
{
  "bench_passes": 5,
  "infer_p50_s": 2.28,
  "infer_p95_s": 2.45,
  "throughput_p50": 14.0,
  "throughput_p95": 13.1
}
```

Optional shape sweep fields (present only when `--bench-shapes` was also passed):

```json
{
  "shapes": [
    {"batch": 1, "seq": 32,  "infer_s": 2.3, "throughput": 13.9},
    {"batch": 1, "seq": 128, "infer_s": 8.7, "throughput": 14.7}
  ]
}
```

## Section 2 — Worker Changes

### Timing separation

`_compile_model()` (`expedition_worker.py:326`) currently measures only `compile_time`. Add an `infer_start`/`infer_time` timer around the `compiled(*inputs)` call. Return signature extends from:

```python
(success, output, compile_time, error, compiled_module)
```

to:

```python
(success, output, compile_time, infer_time, error, compiled_module)
```

All three call sites in the worker are updated. `infer_time` is 0.0 on failure (compile_time already works this way).

### Throughput calculation

Determined by task type, using output shape already available in the decoder:

- **Generative/token tasks** (task in: `text-generation`, `nlp_causal_lm`, `nlp_masked_lm`, `fill-mask`, `nlp_text_cls`, `nlp_token_cls`): `output.shape[1] / infer_s` → `tokens/sec`, where `output.shape[1]` is the output sequence length dimension from the decoded tensor.
- **Everything else** (CV, embeddings, QA, audio, other): `infer_s * 1000` → `ms/sample`

The task-to-unit mapping lives in a small dict in the worker; no dispatcher changes needed.

### Optional bench loop

Activated when `--bench-passes N` (N > 0) is passed to the worker subprocess. Runs after the standard compile+infer while the compiled module is still in scope:

1. Run 2 warm-up passes (times discarded)
2. Run N timed passes, collect `infer_time` for each
3. Compute p50 and p95 from the N samples
4. Derive `throughput_p50` and `throughput_p95` using the same formula as the base pass

The bench loop runs under the existing SIGALRM timeout. If any individual pass times out, the loop stops early and reports whatever passes completed.

### Optional shape sweep

Activated when `--bench-shapes` is set (requires `--bench-passes > 0`; silently ignored otherwise). Varies input shapes after the bench loop completes:

- **LLMs:** re-runs with `seq=[128, 512]` at `batch=1`. Inputs re-constructed by calling `prepare_inputs(seq_len=X)` on the same loader instance.
- **Vision:** re-runs at image size 384 if the standard run used 224.
- **Other tasks:** no shape sweep (single entry in `shapes` list matching the base run).

If a shape fails (OOM, timeout, or any exception) it is skipped silently and not included in the `shapes` list. Shape sweep failure never fails the model.

### perf_history write

New function `append_perf_record(path, record: dict)` added to `bestiary.py`. Opens `data/perf_history.jsonl` in append mode, writes one JSON line, closes. Called from the worker immediately after `bestiary.record_success()`, whether or not bench passes were run.

## Section 3 — CLI Flags and TUI Display

### CLI flags

Added to `expedition.py run` and forwarded to each per-chip worker subprocess (same mechanism as `--timeout-s`):

```
--bench-passes N     Run 2 warm-up + N timed inference passes after each successful
                     compile. Default 0 (disabled). Recommended starting value: 5.

--bench-shapes       Sweep input shapes after bench passes complete.
                     Requires --bench-passes > 0. Silently ignored otherwise.
```

### TUI success line

`_print_success()` updated to show split times and throughput.

Normal run (no bench):
```
compile: 2.8s  infer: 0.4s  13.9 tok/s  pts: +250
compile: 9.8s  infer: 0.1s  103ms/img   pts: +180
```

With bench passes (p50 shown, marked with `~`):
```
compile: 9.8s  infer: 0.1s  ~107ms/img (p50)  pts: +180
```

`total:` is dropped from the display — compile + infer together are now shown separately, so total is redundant.

## What this does not include

- Site changes: the website already reads `best_time_s`; surfacing the new fields is a follow-on once data is flowing.
- Inference-only re-benchmarking of historical models: only newly compiled models get perf records appended.
- Cross-run regression alerting: the JSONL history enables this but it is not built here.
