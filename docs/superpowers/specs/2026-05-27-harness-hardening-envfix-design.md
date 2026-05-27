# Harness Hardening — Env-Fix & Retry Policy

**Date:** 2026-05-27
**Branch:** harness-hardening-envfix
**Goal:** Eliminate harness-caused failures so the reason for failure is almost always forge/XLA, not the compiletron app itself.

## Problem Summary

305 models in the bestiary are marked failed. Analysis shows ~60–70 of those failed for harness/env reasons rather than hardware limitations:

| Root Cause | Count | Category |
|---|---|---|
| cats-image blob missing from HF snapshot | 23 | `other` (ONNX models) |
| JAX models incorrectly permafailed via forge | 21 | `wrong_backend` |
| Stale env version mismatch | 10 | `other` |
| IRD_LF_CACHE env var not set | 13 | `missing_dependency` |

---

## Fix 1: Dataset Pre-warming

**Files:** `lib/expedition/expedition_worker.py`

**Problem:** `huggingface/cats-image` dataset has a parquet index but missing `cats_image.jpeg` blob in the local HF snapshot. All 23 ONNX models that call `load_dataset("huggingface/cats-image")` fail with `FileNotFoundError`.

**Solution:**
1. Add `_warm_hf_datasets()` function called once at expedition startup in `run_expedition()` (before dispatch loop).
2. Function loads `huggingface/cats-image` split="test" and accesses `ds[0]["image"]`. If that raises `FileNotFoundError`, retries with `download_mode="force_redownload"`.
3. On startup, call `bestiary.clear_entries_matching(error_contains="cats_image.jpeg")` to remove the stale entries so they re-run.

**Cost:** ~1s if cache is warm, ~5s on re-download. Runs once per expedition start.

**Recovery:** 23 ONNX models that are sound forge candidates get a fresh shot.

---

## Fix 2: Env Fingerprint + Auto-Reset

**Files:** `lib/expedition/bestiary.py`, `lib/expedition/expedition_worker.py`

**Problem:** 10 models failed because `huggingface-hub==1.15.0` was installed at the time, violating `>=0.30.0,<1.0`. Current env has 0.36.2. These entries are in permafail and will never be retried.

**Solution:**

### 2a. Store env fingerprint on failure
When `bestiary.record_failure()` is called, add `env_fingerprint` to the entry:
```json
{
  "env_fingerprint": {
    "torch": "2.5.1",
    "transformers": "4.52.4",
    "huggingface_hub": "0.36.2"
  }
}
```
Fingerprint is computed once per expedition start and passed through.

### 2b. Auto-clear on version change
At expedition startup, `bestiary.clear_stale_env_failures(current_fingerprint)`:
- Iterates `bestiary.failed`
- For each entry that has `env_fingerprint` AND `error_category in {other, api_mismatch, missing_dependency}` AND `last_error` contains version-mismatch signals (`>=`, `<`, `required`, `version`):
  - If current fingerprint differs from stored fingerprint on any key → delete the entry
- Entries without `env_fingerprint` are not touched (backcompat)

**Effect:** The 10 hub-version entries clear on first run with the new code. New env-caused failures are self-healing when the env is upgraded.

---

## Fix 3: `wrong_backend` Reclassification

**Files:** `lib/expedition/expedition_worker.py`

**Problem:** 21 JAX models reached the forge worker (correctly rejected with `wrong_backend` error) and were recorded as permafail. The router correctly routes them to XLA via priority-1 `jax-native` logic. But because `wrong_backend` is in `_RUNTIME_PERM_FAIL_CATS`, they're locked out permanently.

**Solution:**
1. Remove `wrong_backend` from `_RUNTIME_PERM_FAIL_CATS`.
2. The router's priority-1 rule sends any JAX/Flax model to XLA. On next run they get a real XLA attempt.
3. If XLA is unavailable and they hit `wrong_backend` again, they accumulate normal attempts. At `attempts >= 3`, the regular attempt-count gate catches them.

**Bounce guard:** No special combined-attempt tracking needed — the existing `attempts >= 3` gate already prevents infinite bouncing.

**Effect:** 21 JAX models become retriable. They compile or fail via XLA on merit.

---

## Fix 4: IRD_LF_CACHE Pre-flight

**Files:** `lib/expedition/expedition_worker.py`

**Problem:** 13 seed models require an internal Tenstorrent file server (`IRD_LF_CACHE` env var) to download weights. They fail after starting weight download with `ValueError: IRD_LF_CACHE environment variable is not set`. Wastes time and clutters output.

**Solution:**
Add `_IRD_DEPENDENT_MODEL_PREFIXES` static set to `_preflight_arch_check()`:
```python
_IRD_DEPENDENT_MODEL_PREFIXES = {
    "bevformer", "centernet", "monodepth2", "maptr", "bevdepth",
    "detr3d", "arnold", "fuyu",
}
```
In `_preflight_arch_check()`, before the HF config fetch:
- If `item.model_id.split("/")[0]` is in the set AND `os.environ.get("IRD_LF_CACHE")` is falsy → return `(True, "missing_dependency: IRD_LF_CACHE not set")`
- This check runs for seed models too (not just frontier), so remove the `if not item.is_frontier: return False, ""` guard for this specific check only.

**Effect:** 13 models fail fast (< 1s) with a clean `missing_dependency` instead of mid-download crash.

---

## Bestiary Schema Changes

`bestiary.failed[model_id]` gains one optional new field:
- `env_fingerprint` — dict with `torch`, `transformers`, `huggingface_hub` strings

`bestiary.clear_stale_env_failures()` and `bestiary.clear_entries_matching()` are new methods on the `Bestiary` class.

No changes to existing fields or the compiled section.

---

## Files Changed

| File | Change |
|---|---|
| `lib/expedition/expedition_worker.py` | `_warm_hf_datasets()`, `_IRD_DEPENDENT_MODEL_PREFIXES`, remove `wrong_backend` from permafail set, call warmup + env-reset at startup |
| `lib/expedition/bestiary.py` | `clear_entries_matching()`, `clear_stale_env_failures()`, `record_failure()` gains `env_fingerprint` kwarg |

---

## Success Criteria

1. Running `python3 expedition.py run` with the new code warms the cats-image dataset before dispatch starts.
2. The 10 stale hub-version entries are gone from `bestiary.failed` after startup.
3. The 21 `wrong_backend` models are no longer skipped at the permafail gate; they dispatch to XLA.
4. A model in `_IRD_DEPENDENT_MODEL_PREFIXES` without `IRD_LF_CACHE` set fails immediately with `missing_dependency`, not mid-download.
5. Every failure recorded going forward has an `env_fingerprint` field.
