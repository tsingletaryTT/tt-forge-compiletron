# Harness Hardening — Env-Fix & Retry Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the compiletron harness from being the reason models fail — 60–70 models are currently in permafail for harness/env reasons, not hardware limitations.

**Architecture:** Four targeted fixes to `lib/expedition/bestiary.py` and `lib/expedition/expedition_worker.py`: (1) warm the cats-image HF dataset at startup to fix a missing blob in 23 ONNX loaders; (2) store an env fingerprint on each failure and auto-clear version-mismatch entries when the env is upgraded; (3) remove `wrong_backend` from the permafail category set so 21 JAX models get retried on XLA; (4) add a static IRD_LF_CACHE pre-flight guard so 13 seed models fail fast instead of crashing mid-download.

**Tech Stack:** Python 3.12, `datasets` (HuggingFace), `importlib`, `subprocess`. No new dependencies.

---

## File Map

| File | What changes |
|---|---|
| `lib/expedition/bestiary.py` | New `clear_entries_matching()` method; new `clear_stale_env_failures()` method; `record_failure()` gains `env_fingerprint` kwarg; new `_current_env_fingerprint()` helper |
| `lib/expedition/expedition_worker.py` | New `_warm_hf_datasets()` function; new `_IRD_DEPENDENT_PREFIXES` set; remove `wrong_backend` from `_RUNTIME_PERM_FAIL_CATS`; call warmup + env-reset at top of `run_worker()`; IRD guard added to `_preflight_arch_check()` for seed models |
| `tests/test_bestiary_envfix.py` | New test file: all unit tests for this feature |

---

## Task 1: `clear_entries_matching()` — remove bestiary entries by error substring

**Files:**
- Modify: `lib/expedition/bestiary.py` (after `record_failure`, around line 428)
- Test: `tests/test_bestiary_envfix.py`

- [ ] **Step 1.1: Create the test file with a failing test**

```python
# tests/test_bestiary_envfix.py
import json, tempfile, pathlib, pytest
from lib.expedition.bestiary import Bestiary


def _make_bestiary(tmp_path, failed: dict) -> Bestiary:
    """Write a minimal bestiary JSON and return a loaded Bestiary instance."""
    data = {"compiled": {}, "failed": failed, "chip_totals": {}}
    p = tmp_path / "bestiary.json"
    p.write_text(json.dumps(data))
    return Bestiary(path=p)


def test_clear_entries_matching_removes_matching(tmp_path):
    b = _make_bestiary(tmp_path, {
        "model/a": {"last_error": "FileNotFoundError: cats_image.jpeg", "attempts": 2},
        "model/b": {"last_error": "RuntimeError: segfault", "attempts": 1},
    })
    b.clear_entries_matching(error_contains="cats_image.jpeg")
    assert "model/a" not in b.failed
    assert "model/b" in b.failed


def test_clear_entries_matching_leaves_no_match_untouched(tmp_path):
    b = _make_bestiary(tmp_path, {
        "model/c": {"last_error": "Something else entirely", "attempts": 1},
    })
    b.clear_entries_matching(error_contains="cats_image.jpeg")
    assert "model/c" in b.failed
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 -m pytest tests/test_bestiary_envfix.py::test_clear_entries_matching_removes_matching -xvs 2>&1 | tail -15
```

Expected: `AttributeError: 'Bestiary' object has no attribute 'clear_entries_matching'`

- [ ] **Step 1.3: Add `clear_entries_matching()` to `Bestiary`**

In `lib/expedition/bestiary.py`, after the `record_failure()` method (around line 428):

```python
    def clear_entries_matching(self, *, error_contains: str) -> list[str]:
        """Remove all failed entries whose last_error contains error_contains.

        Returns the list of model_ids that were removed, so callers can log
        what was cleared.  Does not call save() — caller must persist.
        """
        to_remove = [
            mid for mid, entry in self._data["failed"].items()
            if error_contains in entry.get("last_error", "")
        ]
        for mid in to_remove:
            del self._data["failed"][mid]
        return to_remove
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_clear_entries_matching_removes_matching tests/test_bestiary_envfix.py::test_clear_entries_matching_leaves_no_match_untouched -xvs 2>&1 | tail -10
```

Expected: `2 passed`

- [ ] **Step 1.5: Commit**

```bash
git add lib/expedition/bestiary.py tests/test_bestiary_envfix.py
git commit -m "feat: bestiary.clear_entries_matching() for stale-error cleanup"
```

---

## Task 2: Env fingerprint stored on failure + `clear_stale_env_failures()`

**Files:**
- Modify: `lib/expedition/bestiary.py`
- Test: `tests/test_bestiary_envfix.py`

- [ ] **Step 2.1: Add two failing tests**

Append to `tests/test_bestiary_envfix.py`:

```python
def test_record_failure_stores_env_fingerprint(tmp_path):
    b = _make_bestiary(tmp_path, {})
    fingerprint = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    b.record_failure("model/x", run=1, error="ImportError: some version error",
                     env_fingerprint=fingerprint)
    assert b.failed["model/x"]["env_fingerprint"] == fingerprint


def test_clear_stale_env_failures_clears_on_version_change(tmp_path):
    b = _make_bestiary(tmp_path, {
        "model/hub_old": {
            "last_error": "ImportError: huggingface-hub>=0.30.0,<1.0 required but found 1.15.0",
            "error_category": "other",
            "attempts": 3,
            "env_fingerprint": {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "1.15.0"},
        },
        "model/segfault": {
            "last_error": "SIGSEGV: forge.compile() killed by signal 11",
            "error_category": "forge_internal",
            "attempts": 3,
            "env_fingerprint": {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "1.15.0"},
        },
        "model/no_fp": {
            "last_error": "ImportError: version mismatch >= something",
            "error_category": "other",
            "attempts": 2,
            # no env_fingerprint — should be left alone
        },
    })
    current = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    cleared = b.clear_stale_env_failures(current)

    # hub_old: matches — version changed + category + version-signal in error
    assert "model/hub_old" not in b.failed
    assert "model/hub_old" in cleared

    # segfault: forge_internal not in eligible categories — must NOT be cleared
    assert "model/segfault" in b.failed

    # no_fp: no stored fingerprint — must NOT be cleared
    assert "model/no_fp" in b.failed


def test_clear_stale_env_failures_no_change_when_env_same(tmp_path):
    fp = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}
    b = _make_bestiary(tmp_path, {
        "model/y": {
            "last_error": "ImportError: version >= 1.0 required",
            "error_category": "api_mismatch",
            "attempts": 2,
            "env_fingerprint": fp,
        },
    })
    cleared = b.clear_stale_env_failures(fp)  # same fingerprint
    assert "model/y" in b.failed
    assert cleared == []
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_record_failure_stores_env_fingerprint tests/test_bestiary_envfix.py::test_clear_stale_env_failures_clears_on_version_change -xvs 2>&1 | tail -10
```

Expected: `TypeError` on `record_failure` unexpected kwarg.

- [ ] **Step 2.3: Add `_current_env_fingerprint()`, update `record_failure()`, add `clear_stale_env_failures()`**

In `lib/expedition/bestiary.py`, add after the module-level imports (before `class Bestiary:`):

```python
def _current_env_fingerprint() -> dict[str, str]:
    """Return a dict of key library versions installed in the current Python env.

    Used to detect when the environment has changed since a failure was recorded,
    so those failures can be auto-cleared on the next run.
    """
    result: dict[str, str] = {}
    for pkg, attr in (
        ("torch",           "__version__"),
        ("transformers",    "__version__"),
        ("huggingface_hub", "__version__"),
    ):
        try:
            import importlib
            mod = importlib.import_module(pkg)
            result[pkg] = getattr(mod, attr, "unknown")
        except ImportError:
            result[pkg] = "not_installed"
    return result
```

Update `record_failure()` signature and body in `lib/expedition/bestiary.py`:

```python
    def record_failure(self, model_id: str, run: int, error: str,
                       env_fingerprint: dict[str, str] | None = None) -> None:
        """Track a failed compilation attempt for retry-interest purposes.

        Failures are stored separately from compiled models. They are NOT used
        for score penalties (the -10 penalty is applied by the caller at runtime),
        but they inform future run decisions (e.g. skip models that always fail).

        Args:
            model_id:        HuggingFace model identifier.
            run:             Sequential run number within the current expedition session.
            error:           String representation of the exception or error message.
            env_fingerprint: Optional dict from _current_env_fingerprint(). When provided,
                             stored on the entry so clear_stale_env_failures() can detect
                             environment upgrades.
        """
        if model_id not in self._data["failed"]:
            self._data["failed"][model_id] = {
                "run_first_failed": run,
                "attempts": 0,
                "last_error": "",
                "error_category": "",
            }
        entry = self._data["failed"][model_id]
        entry["attempts"] += 1
        entry["last_error"] = error
        # Always recompute so the category stays in sync if the error changes.
        entry["error_category"] = _classify_error(error)[0]
        if env_fingerprint is not None:
            entry["env_fingerprint"] = env_fingerprint
```

Add `clear_stale_env_failures()` after `clear_entries_matching()`:

```python
    def clear_stale_env_failures(self, current_fingerprint: dict[str, str]) -> list[str]:
        """Remove failed entries whose environment has changed since failure was recorded.

        Only clears entries that meet ALL THREE conditions:
          1. Have a stored env_fingerprint that differs from current_fingerprint.
          2. Have error_category in {other, api_mismatch, missing_dependency}.
          3. Have a version-signal keyword in last_error (>=, <=, <, >, required, version).

        This prevents clearing hardware failures (forge_internal, SIGSEGV) that
        happen to have env fingerprints, while clearing genuine env-version errors.

        Returns the list of model_ids removed.  Caller must call save().
        """
        _ELIGIBLE_CATS = {"other", "api_mismatch", "missing_dependency"}
        _VERSION_SIGNALS = (">=", "<=", "<", ">", "required", "version")

        to_remove = []
        for mid, entry in self._data["failed"].items():
            stored_fp = entry.get("env_fingerprint")
            if not stored_fp:
                continue
            if stored_fp == current_fingerprint:
                continue
            if entry.get("error_category") not in _ELIGIBLE_CATS:
                continue
            err = entry.get("last_error", "")
            if not any(sig in err for sig in _VERSION_SIGNALS):
                continue
            to_remove.append(mid)

        for mid in to_remove:
            del self._data["failed"][mid]
        return to_remove
```

- [ ] **Step 2.4: Run all four new tests**

```bash
python3 -m pytest tests/test_bestiary_envfix.py -xvs 2>&1 | tail -15
```

Expected: `6 passed` (2 from Task 1 + 4 new)

- [ ] **Step 2.5: Commit**

```bash
git add lib/expedition/bestiary.py tests/test_bestiary_envfix.py
git commit -m "feat: env fingerprint on failures + clear_stale_env_failures() auto-reset"
```

---

## Task 3: Dataset pre-warming + startup env-reset in `run_worker()`

**Files:**
- Modify: `lib/expedition/expedition_worker.py`
- Test: `tests/test_bestiary_envfix.py`

- [ ] **Step 3.1: Add a failing test for `_warm_hf_datasets()`**

Append to `tests/test_bestiary_envfix.py`:

```python
def test_warm_hf_datasets_is_importable():
    """_warm_hf_datasets must be importable without running forge or hardware."""
    from lib.expedition.expedition_worker import _warm_hf_datasets
    assert callable(_warm_hf_datasets)
```

- [ ] **Step 3.2: Run test to confirm it fails**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_warm_hf_datasets_is_importable -xvs 2>&1 | tail -10
```

Expected: `ImportError: cannot import name '_warm_hf_datasets'`

- [ ] **Step 3.3: Add `_warm_hf_datasets()` to `expedition_worker.py`**

Add this function near the top of `lib/expedition/expedition_worker.py`, after the `_CUSTOM_DEP_MAP` block (around line 658):

```python
def _warm_hf_datasets(bestiary: "Bestiary") -> None:
    """Ensure the huggingface/cats-image dataset blob is present in the local cache.

    Many ONNX loaders in tt-forge-models call load_dataset("huggingface/cats-image")
    and directly access the image blob.  The HF hub may have the parquet index cached
    but the image blob missing, causing FileNotFoundError mid-compile.  One access of
    ds[0]["image"] populates the blob; force_redownload recovers a broken cache.

    Also auto-clears any bestiary entries that failed with the cats_image.jpeg error,
    since those models are sound forge candidates now that the cache is repaired.
    """
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("huggingface/cats-image", split="test")
            _ = ds[0]["image"]  # triggers blob download if missing
        except FileNotFoundError:
            ds = load_dataset("huggingface/cats-image", split="test",
                              download_mode="force_redownload")
            _ = ds[0]["image"]
        cleared = bestiary.clear_entries_matching(error_contains="cats_image.jpeg")
        if cleared:
            print(f"  {GREEN}✓ dataset warm-up cleared {len(cleared)} stale cats-image entries{RESET}")
            bestiary.save()
    except Exception as exc:
        # Fail open — a missing dataset must never block the expedition.
        print(f"  {YELLOW}⚠ dataset warm-up skipped: {exc}{RESET}")
```

- [ ] **Step 3.4: Run test to confirm import works**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_warm_hf_datasets_is_importable -xvs 2>&1 | tail -10
```

Expected: `1 passed`

- [ ] **Step 3.5: Wire `_warm_hf_datasets()` and env-reset into `run_worker()` startup**

In `lib/expedition/expedition_worker.py`, in `run_worker()` — find the block that prints the expedition header banner (the `═══` line, around line 1553). Insert the startup calls **after** `hud.write_status()` and **before** the `results: list[dict] = []` line:

```python
    # ── Startup pre-flight: dataset cache warmup + stale env entry cleanup ──
    # Warm the cats-image dataset blob before any model dispatch so ONNX loaders
    # that call load_dataset("huggingface/cats-image") don't hit FileNotFoundError.
    # Also auto-clear bestiary entries that failed for version-mismatch reasons
    # when the current env is newer than the env recorded at failure time.
    from lib.expedition.bestiary import _current_env_fingerprint
    _env_fp = _current_env_fingerprint()
    _warm_hf_datasets(bestiary)
    _stale_cleared = bestiary.clear_stale_env_failures(_env_fp)
    if _stale_cleared:
        print(f"  {GREEN}✓ env upgrade cleared {len(_stale_cleared)} stale failure entries{RESET}")
        bestiary.save()
```

- [ ] **Step 3.6: Run the full test suite to confirm nothing broken**

```bash
python3 -m pytest tests/ -x --ignore=tests/lib -q 2>&1 | tail -20
```

Expected: all existing tests still pass, plus the new ones.

- [ ] **Step 3.7: Commit**

```bash
git add lib/expedition/expedition_worker.py tests/test_bestiary_envfix.py
git commit -m "feat: dataset pre-warm + env-reset at expedition startup"
```

---

## Task 4: Remove `wrong_backend` from permafail categories

**Files:**
- Modify: `lib/expedition/expedition_worker.py` (line ~1639)
- Test: `tests/test_bestiary_envfix.py`

- [ ] **Step 4.1: Add a failing test**

Append to `tests/test_bestiary_envfix.py`:

```python
def test_wrong_backend_not_in_perm_fail_cats():
    """wrong_backend must not be in _RUNTIME_PERM_FAIL_CATS so JAX models get retried."""
    import ast, pathlib
    src = pathlib.Path("lib/expedition/expedition_worker.py").read_text()
    # Find the set literal assigned to _RUNTIME_PERM_FAIL_CATS
    # Quick textual check is reliable enough here
    assert "wrong_backend" not in src.split("_RUNTIME_PERM_FAIL_CATS")[1].split("}")[0], \
        "wrong_backend must not appear inside _RUNTIME_PERM_FAIL_CATS"
```

- [ ] **Step 4.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_wrong_backend_not_in_perm_fail_cats -xvs 2>&1 | tail -10
```

Expected: `AssertionError: wrong_backend must not appear inside _RUNTIME_PERM_FAIL_CATS`

- [ ] **Step 4.3: Remove `wrong_backend` from `_RUNTIME_PERM_FAIL_CATS`**

In `lib/expedition/expedition_worker.py`, find `_RUNTIME_PERM_FAIL_CATS` (around line 1639):

```python
        _RUNTIME_PERM_FAIL_CATS = {
            "forge_internal", "unsupported_arch", "loader_missing",
            "missing_dependency", "unsupported_backend", "xla_runtime_error",
            "api_mismatch", "shape_mismatch", "forge_missing_op",
            "model_access", "wrong_backend", "model_bug",
        }
```

Change to:

```python
        _RUNTIME_PERM_FAIL_CATS = {
            "forge_internal", "unsupported_arch", "loader_missing",
            "missing_dependency", "unsupported_backend", "xla_runtime_error",
            "api_mismatch", "shape_mismatch", "forge_missing_op",
            "model_access", "model_bug",
            # wrong_backend intentionally excluded: JAX models reaching the forge
            # worker should retry — the router re-routes them to XLA on next attempt.
            # The standard attempts>=3 gate still catches persistent bounce loops.
        }
```

- [ ] **Step 4.4: Run test to confirm it passes**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_wrong_backend_not_in_perm_fail_cats -xvs 2>&1 | tail -10
```

Expected: `1 passed`

- [ ] **Step 4.5: Run full suite**

```bash
python3 -m pytest tests/ -x --ignore=tests/lib -q 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 4.6: Commit**

```bash
git add lib/expedition/expedition_worker.py tests/test_bestiary_envfix.py
git commit -m "fix: remove wrong_backend from permafail categories — JAX models retry via XLA"
```

---

## Task 5: IRD_LF_CACHE pre-flight guard

**Files:**
- Modify: `lib/expedition/expedition_worker.py`
- Test: `tests/test_bestiary_envfix.py`

- [ ] **Step 5.1: Add a failing test**

Append to `tests/test_bestiary_envfix.py`:

```python
def test_ird_preflight_symbols_exist():
    """_IRD_DEPENDENT_PREFIXES must exist and contain known IRD models."""
    from lib.expedition.expedition_worker import _IRD_DEPENDENT_PREFIXES
    assert "bevformer" in _IRD_DEPENDENT_PREFIXES
    assert "centernet" in _IRD_DEPENDENT_PREFIXES
    assert "yolov3" in _IRD_DEPENDENT_PREFIXES
```

- [ ] **Step 5.2: Run to confirm it fails**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_ird_preflight_symbols_exist -xvs 2>&1 | tail -10
```

Expected: `ImportError: cannot import name '_IRD_DEPENDENT_PREFIXES'`

- [ ] **Step 5.3: Add `_IRD_DEPENDENT_PREFIXES` constant and guard to `_preflight_arch_check()`**

In `lib/expedition/expedition_worker.py`, add the constant immediately before `_preflight_arch_check()` (around line 658, after `_CUSTOM_DEP_MAP`):

```python
# Seed model ID prefixes whose loaders call tt-forge-models/tools/utils.py _get_file(),
# which requires the IRD_LF_CACHE env var (a Tenstorrent-internal file server).
# Without it these models download large weights then crash; fail fast instead.
_IRD_DEPENDENT_PREFIXES: frozenset[str] = frozenset({
    "bevformer",
    "centernet",
    "maptr",
    "monodepth2",
    "mplug_owl2",
    "petr",
    "ssd512",
    "ultra_fast_lane_detection_v2",
    "whisper",     # audio_classification/onnx variant uses IRD cache
    "yolov3",
    "yolov4",
})
```

Then, in `_preflight_arch_check()`, replace the opening guard:

```python
    if not item.is_frontier:
        return False, ""
```

with:

```python
    # IRD_LF_CACHE guard applies to seed models too (not just frontier).
    # Check this before the frontier-only return below.
    model_prefix = item.model_id.split("/")[0]
    if model_prefix in _IRD_DEPENDENT_PREFIXES and not os.environ.get("IRD_LF_CACHE"):
        return (True,
                f"missing_dependency: IRD_LF_CACHE env var not set "
                f"(model '{item.model_id}' requires Tenstorrent internal file cache)")

    if not item.is_frontier:
        return False, ""
```

- [ ] **Step 5.4: Run test to confirm it passes**

```bash
python3 -m pytest tests/test_bestiary_envfix.py::test_ird_preflight_symbols_exist -xvs 2>&1 | tail -10
```

Expected: `1 passed`

- [ ] **Step 5.5: Run full suite**

```bash
python3 -m pytest tests/ -x --ignore=tests/lib -q 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 5.6: Commit**

```bash
git add lib/expedition/expedition_worker.py tests/test_bestiary_envfix.py
git commit -m "feat: IRD_LF_CACHE pre-flight guard — fail fast for 13 internal-server models"
```

---

## Task 6: Wire `env_fingerprint` into every `record_failure()` call site

**Files:**
- Modify: `lib/expedition/expedition_worker.py`

The fingerprint is only useful if it's stored. All call sites in `run_worker()` need to pass it.

- [ ] **Step 6.1: Find all `record_failure()` call sites in `expedition_worker.py`**

```bash
grep -n "bestiary.record_failure" lib/expedition/expedition_worker.py
```

Note the line numbers. There will be several (gated check, arch check, main compile result, etc.).

- [ ] **Step 6.2: Add `_env_fp` to every `bestiary.record_failure()` call in `run_worker()`**

The `_env_fp` variable is already computed in Task 3's startup block. Each `bestiary.record_failure(...)` call in `run_worker()` needs `env_fingerprint=_env_fp` added as a kwarg.

For each call site found in Step 6.1 that lives inside `run_worker()`, change:
```python
bestiary.record_failure(item.model_id, run_number, some_err)
```
to:
```python
bestiary.record_failure(item.model_id, run_number, some_err, env_fingerprint=_env_fp)
```

Do the same for the arch-preflight call site and gated-check call site inside `run_worker()`.

- [ ] **Step 6.3: Run full suite to confirm nothing broken**

```bash
python3 -m pytest tests/ -x --ignore=tests/lib -q 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 6.4: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "feat: pass env_fingerprint to all record_failure() calls in run_worker"
```

---

## Task 7: Smoke test — verify startup warmup and env-reset run cleanly

**Files:**
- Test: `tests/test_bestiary_envfix.py`

This is an integration-style test that exercises the full startup sequence against a controlled bestiary.

- [ ] **Step 7.1: Add integration smoke test**

Append to `tests/test_bestiary_envfix.py`:

```python
def test_warm_hf_datasets_does_not_crash(tmp_path):
    """_warm_hf_datasets must not raise even if datasets library behaves unexpectedly."""
    import unittest.mock as mock
    from lib.expedition.expedition_worker import _warm_hf_datasets
    from lib.expedition.bestiary import Bestiary

    b = _make_bestiary(tmp_path, {
        "model/onnx": {
            "last_error": "FileNotFoundError: cats_image.jpeg missing",
            "error_category": "other",
            "attempts": 2,
        }
    })

    # Simulate successful dataset load (blob available after load)
    fake_img = mock.MagicMock()
    fake_ds = mock.MagicMock()
    fake_ds.__getitem__ = mock.MagicMock(return_value={"image": fake_img})

    with mock.patch("datasets.load_dataset", return_value=fake_ds):
        _warm_hf_datasets(b)

    # Entry should be cleared because last_error contains cats_image.jpeg
    assert "model/onnx" not in b.failed


def test_env_reset_clears_hub_version_entries(tmp_path):
    """Stale hub-version entries are cleared when env is upgraded."""
    from lib.expedition.bestiary import Bestiary

    old_fp = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "1.15.0"}
    new_fp = {"torch": "2.5.1", "transformers": "4.52.4", "huggingface_hub": "0.36.2"}

    b = _make_bestiary(tmp_path, {
        "albert/albert-base-v1": {
            "last_error": "ImportError: huggingface-hub>=0.30.0,<1.0 required but found 1.15.0",
            "error_category": "other",
            "attempts": 3,
            "env_fingerprint": old_fp,
        }
    })

    cleared = b.clear_stale_env_failures(new_fp)
    assert "albert/albert-base-v1" in cleared
    assert "albert/albert-base-v1" not in b.failed
```

- [ ] **Step 7.2: Run all tests in the new file**

```bash
python3 -m pytest tests/test_bestiary_envfix.py -v 2>&1 | tail -25
```

Expected: all 11 tests pass.

- [ ] **Step 7.3: Run the full test suite**

```bash
python3 -m pytest tests/ --ignore=tests/lib -q 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 7.4: Commit**

```bash
git add tests/test_bestiary_envfix.py
git commit -m "test: integration smoke tests for startup warmup and env-reset"
```

---

## Task 8: Update CLAUDE.md and close the branch

- [ ] **Step 8.1: Add a summary entry to CLAUDE.md**

In `/home/ttuser/code/tt-forge-compiletron/CLAUDE.md`, append under a new section:

```markdown
## Harness Hardening — Env-Fix (2026-05-27)

**Goal:** Stop the harness from being the failure reason. ~60 models were in permafail for env/infra reasons.

**What was fixed:**
1. **Dataset pre-warm** — `_warm_hf_datasets()` runs at expedition startup, repairs the `huggingface/cats-image` blob symlink, clears 23 stale ONNX entries.
2. **Env fingerprint + auto-reset** — `record_failure()` stores torch/transformers/hub versions; `clear_stale_env_failures()` auto-removes version-mismatch entries when env is upgraded.
3. **wrong_backend** removed from `_RUNTIME_PERM_FAIL_CATS` — 21 JAX models retry via XLA instead of staying locked out permanently.
4. **IRD_LF_CACHE pre-flight** — `_IRD_DEPENDENT_PREFIXES` frozenset lets 13 seed models fail fast (< 1s) instead of downloading weights then crashing.

**Key files:** `lib/expedition/bestiary.py`, `lib/expedition/expedition_worker.py`, `tests/test_bestiary_envfix.py`
```

- [ ] **Step 8.2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with harness-hardening-envfix summary"
```

- [ ] **Step 8.3: Verify branch is clean**

```bash
git status
git log --oneline main..HEAD
```

Expected: clean working tree; ~8 commits ahead of main.

---

## Self-Review Checklist

**Spec coverage:**
- Fix 1 (cats-image pre-warm) → Task 3 ✓
- Fix 2 (env fingerprint + auto-reset) → Tasks 2 + 3 + 6 ✓
- Fix 3 (wrong_backend reclassification) → Task 4 ✓
- Fix 4 (IRD_LF_CACHE pre-flight) → Task 5 ✓
- Bestiary schema changes (`clear_entries_matching`, `clear_stale_env_failures`, `env_fingerprint`) → Tasks 1 + 2 ✓

**Type consistency across tasks:**
- `clear_entries_matching(*, error_contains: str) -> list[str]` — defined Task 1, called Task 3 ✓
- `clear_stale_env_failures(current_fingerprint: dict[str, str]) -> list[str]` — defined Task 2, called Task 3 ✓
- `record_failure(..., env_fingerprint: dict[str, str] | None = None)` — defined Task 2, wired Task 6 ✓
- `_current_env_fingerprint() -> dict[str, str]` — defined Task 2, imported in Task 3 call site ✓
- `_warm_hf_datasets(bestiary: Bestiary) -> None` — defined Task 3, callable assertion Task 3 ✓
- `_IRD_DEPENDENT_PREFIXES: frozenset[str]` — defined Task 5, imported in test ✓

**No placeholders:** All code blocks are complete. ✓
