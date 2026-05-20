# Ephemeral Cache Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--ephemeral` and `--evict-failures` flags to the expedition that evict freshly-downloaded HuggingFace model weights after each compile+bench cycle, preserving disk space across long runs while keeping gold-star models.

**Architecture:** A new `cache_janitor` module owns all HF cache logic (snapshot, gold-star decision, eviction). Both worker processes call into it after each model result. The orchestrator forwards the flags via env vars to the shell launcher, which passes them to the worker CLI.

**Tech Stack:** `huggingface_hub.scan_cache_dir`, `shutil.rmtree`, existing `ScoreResult` / `Rarity` from `scorer.py`, `argparse` in both workers and `expedition.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `lib/expedition/cache_janitor.py` | Create | HF cache snapshot, gold-star check, eviction logic |
| `tests/lib/test_cache_janitor.py` | Create | Unit tests for all three public functions |
| `lib/expedition/expedition_worker.py` | Modify | Add CLI args, `run_worker()` params, startup snapshot, post-result eviction |
| `lib/expedition/expedition_worker_xla.py` | Modify | Same pattern as above for XLA path |
| `scripts/run_expedition.sh` | Modify | Read `EXPEDITION_EPHEMERAL` / `EXPEDITION_EVICT_FAILURES` env vars, forward to workers |
| `expedition.py` | Modify | Add `--ephemeral` / `--evict-failures` CLI flags, set env vars before subprocess |

---

## Task 1: `cache_janitor.py` module

**Files:**
- Create: `lib/expedition/cache_janitor.py`
- Test: `tests/lib/test_cache_janitor.py`

- [ ] **Step 1: Write failing tests for `snapshot_preexisting`**

Create `tests/lib/test_cache_janitor.py`:

```python
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from lib.expedition.cache_janitor import snapshot_preexisting, is_gold_star, maybe_evict
from lib.expedition.scorer import ScoreResult, Rarity, Newness


def _score(pts: int, rarity=Rarity.COMMON, first_ever=False) -> ScoreResult:
    return ScoreResult(
        pts=pts, is_first_ever=first_ever, rarity=rarity,
        newness=Newness.ESTABLISHED, streak_at_score=0,
    )


class TestSnapshotPreexisting:
    def test_returns_frozenset_of_repo_ids(self):
        mock_repo = MagicMock()
        mock_repo.repo_id = "openai-community/gpt2"
        mock_info = MagicMock()
        mock_info.repos = [mock_repo]
        with patch("lib.expedition.cache_janitor.scan_cache_dir", return_value=mock_info):
            result = snapshot_preexisting()
        assert result == frozenset({"openai-community/gpt2"})

    def test_returns_empty_on_scan_error(self):
        with patch("lib.expedition.cache_janitor.scan_cache_dir", side_effect=Exception("no cache")):
            result = snapshot_preexisting()
        assert result == frozenset()

    def test_returns_frozenset_type(self):
        mock_info = MagicMock()
        mock_info.repos = []
        with patch("lib.expedition.cache_janitor.scan_cache_dir", return_value=mock_info):
            result = snapshot_preexisting()
        assert isinstance(result, frozenset)
```

- [ ] **Step 2: Run to confirm failures**

```bash
python3 -m pytest tests/lib/test_cache_janitor.py -v 2>&1 | tail -15
```

Expected: `ImportError` or `ModuleNotFoundError` — `cache_janitor.py` does not exist yet.

- [ ] **Step 3: Write failing tests for `is_gold_star`**

Append to `tests/lib/test_cache_janitor.py`:

```python
class TestIsGoldStar:
    def test_legendary_success_is_gold(self):
        assert is_gold_star(_score(400, rarity=Rarity.LEGENDARY)) is True

    def test_rare_success_is_gold(self):
        assert is_gold_star(_score(300, rarity=Rarity.RARE)) is True

    def test_first_ever_success_is_gold(self):
        assert is_gold_star(_score(250, rarity=Rarity.COMMON, first_ever=True)) is True

    def test_common_success_not_gold(self):
        assert is_gold_star(_score(100, rarity=Rarity.COMMON)) is False

    def test_uncommon_success_not_gold(self):
        assert is_gold_star(_score(150, rarity=Rarity.UNCOMMON)) is False

    def test_failure_never_gold(self):
        # Even legendary rarity: a failure (pts <= 0) is not gold star
        assert is_gold_star(_score(-10, rarity=Rarity.LEGENDARY, first_ever=True)) is False

    def test_zero_pts_not_gold(self):
        assert is_gold_star(_score(0, rarity=Rarity.RARE)) is False
```

- [ ] **Step 4: Write failing tests for `maybe_evict`**

Append to `tests/lib/test_cache_janitor.py`:

```python
class TestMaybeEvict:
    def test_skips_preexisting_model(self, tmp_path):
        preexisting = frozenset({"org/model"})
        evicted, freed = maybe_evict("org/model", _score(100), preexisting)
        assert evicted is False
        assert freed == 0

    def test_skips_if_no_hf_cache_entry(self, tmp_path):
        preexisting = frozenset()
        # No cache directory exists for this model
        with patch("lib.expedition.cache_janitor._hf_repo_dir",
                   return_value=tmp_path / "nonexistent"):
            evicted, freed = maybe_evict("org/model", _score(100), preexisting)
        assert evicted is False

    def test_evicts_successful_common_model(self, tmp_path):
        repo_dir = tmp_path / "models--org--model"
        repo_dir.mkdir()
        (repo_dir / "weights.bin").write_bytes(b"x" * 1024)

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir") as mock_scan:
            mock_repo = MagicMock()
            mock_repo.repo_id = "org/model"
            mock_repo.size_on_disk = 1024
            mock_scan.return_value.repos = [mock_repo]

            evicted, freed = maybe_evict("org/model", _score(100, rarity=Rarity.COMMON),
                                         frozenset())
        assert evicted is True
        assert freed == 1024
        assert not repo_dir.exists()

    def test_preserves_gold_star_model(self, tmp_path):
        repo_dir = tmp_path / "models--org--bigmodel"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir):
            evicted, freed = maybe_evict(
                "org/bigmodel",
                _score(400, rarity=Rarity.LEGENDARY),
                frozenset(),
            )
        assert evicted is False
        assert repo_dir.exists()

    def test_keeps_failure_without_evict_failures_flag(self, tmp_path):
        repo_dir = tmp_path / "models--org--failmodel"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir):
            evicted, freed = maybe_evict(
                "org/failmodel", _score(-10), frozenset(), evict_failures=False
            )
        assert evicted is False
        assert repo_dir.exists()

    def test_evicts_failure_with_evict_failures_flag(self, tmp_path):
        repo_dir = tmp_path / "models--org--failmodel"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir") as mock_scan:
            mock_scan.return_value.repos = []  # size unknown → 0

            evicted, freed = maybe_evict(
                "org/failmodel", _score(-10), frozenset(), evict_failures=True
            )
        assert evicted is True
        assert not repo_dir.exists()

    def test_survives_rmtree_error(self, tmp_path):
        repo_dir = tmp_path / "models--org--model"
        repo_dir.mkdir()

        with patch("lib.expedition.cache_janitor._hf_repo_dir", return_value=repo_dir), \
             patch("lib.expedition.cache_janitor.scan_cache_dir") as mock_scan, \
             patch("shutil.rmtree", side_effect=OSError("permission denied")):
            mock_scan.return_value.repos = []
            evicted, freed = maybe_evict("org/model", _score(100), frozenset())
        assert evicted is False
```

- [ ] **Step 5: Run all tests to confirm failures**

```bash
python3 -m pytest tests/lib/test_cache_janitor.py -v 2>&1 | tail -20
```

Expected: All fail with `ImportError`.

- [ ] **Step 6: Implement `lib/expedition/cache_janitor.py`**

Create `lib/expedition/cache_janitor.py`:

```python
"""HuggingFace cache janitor for ephemeral expedition runs.

Tracks which model repos were pre-existing before the run, decides whether
a result earns gold-star preservation, and evicts net-new downloads when
the model doesn't qualify.

Public API:
    snapshot_preexisting() -> frozenset[str]
    is_gold_star(result: ScoreResult) -> bool
    maybe_evict(model_id, result, preexisting, evict_failures=False) -> tuple[bool, int]
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def snapshot_preexisting() -> frozenset[str]:
    """Return the set of HF model repo IDs already in cache before this run.

    Called once at worker startup. Returns an empty frozenset on any error so
    the expedition continues — the conservative path is to treat everything as
    pre-existing (nothing gets evicted) rather than aborting the run.
    """
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return frozenset(repo.repo_id for repo in info.repos)
    except Exception as exc:
        log.warning("cache_janitor: snapshot failed (%s); treating all as pre-existing", exc)
        return frozenset()


def is_gold_star(result) -> bool:
    """Return True if a successful result earns gold-star preservation.

    Gold star = pts > 0 AND (rarity rare/legendary OR first-ever compile).
    Failures (pts <= 0) are never gold star even if rarity is legendary.
    """
    if result.pts <= 0:
        return False
    return result.rarity in ("rare", "legendary") or result.is_first_ever


def _hf_repo_dir(model_id: str) -> Path:
    """Map a HuggingFace model_id to its local cache directory path."""
    safe = model_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe}"


def maybe_evict(
    model_id: str,
    result,
    preexisting: frozenset[str],
    evict_failures: bool = False,
) -> tuple[bool, int]:
    """Evict model weights from HF cache if appropriate.

    Only acts on net-new downloads (not in preexisting). Returns
    (evicted, bytes_freed). Never raises — logs and returns (False, 0) on
    any error so the expedition is never aborted by cleanup failures.

    Args:
        model_id:       HuggingFace model identifier, e.g. "openai-community/gpt2".
        result:         ScoreResult from scorer.compute_score().
        preexisting:    frozenset of repo_id strings present before this run.
        evict_failures: If True, also evict weights for failed models.
    """
    # Pre-existing cache entries are never touched.
    if model_id in preexisting:
        return False, 0

    repo_dir = _hf_repo_dir(model_id)
    if not repo_dir.exists():
        return False, 0  # static/local loader — no HF cache entry

    success = result.pts > 0
    if success:
        if is_gold_star(result):
            return False, 0  # gold star: preserve
    else:
        if not evict_failures:
            return False, 0  # keep failures unless explicitly asked

    # Measure size before deletion (best-effort).
    bytes_freed = 0
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        repo_info = next((r for r in info.repos if r.repo_id == model_id), None)
        if repo_info:
            bytes_freed = repo_info.size_on_disk
    except Exception:
        pass  # size unknown; evict anyway

    try:
        shutil.rmtree(repo_dir)
        return True, bytes_freed
    except Exception as exc:
        log.warning("cache_janitor: eviction failed for %s: %s", model_id, exc)
        return False, 0
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
python3 -m pytest tests/lib/test_cache_janitor.py -v 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] **Step 8: Run full test suite**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: `261 passed` (or more if other tests exist).

- [ ] **Step 9: Commit**

```bash
git add lib/expedition/cache_janitor.py tests/lib/test_cache_janitor.py
git commit -m "feat: add cache_janitor module for ephemeral weight eviction"
```

---

## Task 2: Integrate into `expedition_worker.py`

**Files:**
- Modify: `lib/expedition/expedition_worker.py`
  - `run_worker()` signature: line ~997
  - arg parser: lines ~1316–1337
  - startup snapshot: after items are loaded (~line 1056)
  - post-result eviction: after `bestiary.save()` at line 1274

- [ ] **Step 1: Add `ephemeral` / `evict_failures` to `run_worker()` signature**

Find the `def run_worker(` line (~997) and update the signature:

```python
def run_worker(chip_id: int, run_number: int, bestiary_path: str,
               queue_path: str | None, results_path: str,
               model_json_path: str | None = None,
               bench_passes: int = 0,
               bench_shapes: bool = False,
               ephemeral: bool = False,
               evict_failures: bool = False) -> None:
```

Also add the two new params to the docstring Args block (after `bench_shapes`):

```
        ephemeral:       If True, evict net-new HF downloads after each model unless gold star.
        evict_failures:  If True and ephemeral, also evict weights for failed models.
```

- [ ] **Step 2: Add preexisting snapshot at worker startup**

Find the `results: list[dict] = []` line (~1056) and insert immediately after it:

```python
    # Snapshot cache before any downloads so we only evict what this run fetched.
    preexisting: frozenset[str] = frozenset()
    if ephemeral:
        from lib.expedition import cache_janitor as _janitor
        preexisting = _janitor.snapshot_preexisting()
```

- [ ] **Step 3: Add eviction call after `bestiary.save()`**

Find the `bestiary.save()` line (~1274) and insert after it (before `hud.write_status()`):

```python
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
```

Note: `success` is defined at line ~1113 from the `_compile_model()` return, and `score` is always defined by this point (success path ~1144, failure path ~1265). `DIM`, `RESET`, `GOLD` are module-level constants already defined at lines 82–91.

- [ ] **Step 4: Add `--ephemeral` and `--evict-failures` to the arg parser**

Find the `parser.add_argument("--bench-shapes"` block (~1334) and add after it:

```python
    parser.add_argument("--ephemeral", action="store_true",
                        help="Evict net-new HF model weights after each result "
                             "unless the model earns a gold-star rating.")
    parser.add_argument("--evict-failures", action="store_true",
                        help="With --ephemeral, also evict weights for failed models.")
```

- [ ] **Step 5: Forward new args into `run_worker()` call**

Find the `run_worker(` call at ~line 1343 and add the two new kwargs:

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
        ephemeral=args.ephemeral,
        evict_failures=args.evict_failures,
    )
```

- [ ] **Step 6: Verify the worker accepts the new flags without error**

```bash
python3 lib/expedition/expedition_worker.py --help 2>&1 | grep -A2 "ephemeral\|evict"
```

Expected output includes:
```
  --ephemeral           Evict net-new HF model weights after each result...
  --evict-failures      With --ephemeral, also evict weights for failed models.
```

- [ ] **Step 7: Run full test suite**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "feat: wire --ephemeral/--evict-failures into forge expedition worker"
```

---

## Task 3: Integrate into `expedition_worker_xla.py`

**Files:**
- Modify: `lib/expedition/expedition_worker_xla.py`
  - `run_worker_xla()` signature: line ~1032
  - arg parser: lines ~1345–1371
  - startup snapshot and post-result eviction (same pattern as Task 2)

- [ ] **Step 1: Add `ephemeral` / `evict_failures` to `run_worker_xla()` signature**

Find `def run_worker_xla(` (~line 1032) and update:

```python
def run_worker_xla(chip_id: int, run_number: int, bestiary_path: str,
                   queue_path: str | None, results_path: str,
                   model_json_path: str | None = None,
                   bench_passes: int = 0,
                   bench_shapes: bool = False,
                   ephemeral: bool = False,
                   evict_failures: bool = False) -> None:
```

Add to the docstring Args block (after `bench_shapes`):

```
        ephemeral:       If True, evict net-new HF downloads after each model unless gold star.
        evict_failures:  If True and ephemeral, also evict weights for failed models.
```

- [ ] **Step 2: Find where XLA model results are finalized**

Run:

```bash
grep -n "bestiary.save\|record_failure\|record_success\|hud.write_status" lib/expedition/expedition_worker_xla.py | tail -20
```

Note the line numbers — you'll insert the snapshot and eviction calls at the same structural positions as in Task 2 (snapshot after `results: list = []`, eviction after `bestiary.save()`).

- [ ] **Step 3: Add preexisting snapshot at worker startup**

Find `results: list` initialization in `run_worker_xla()` and insert after it:

```python
    preexisting: frozenset[str] = frozenset()
    if ephemeral:
        from lib.expedition import cache_janitor as _janitor
        preexisting = _janitor.snapshot_preexisting()
```

- [ ] **Step 4: Add eviction call after `bestiary.save()` in XLA worker**

After each `bestiary.save()` in the XLA model loop, insert the same block as Task 2 Step 3. The XLA worker uses the same `score`, `success`, `item.model_id` variable names. Check whether `DIM`, `RESET`, `GOLD` constants exist in `expedition_worker_xla.py`:

```bash
grep -n "^GOLD\|^DIM\|^RESET" lib/expedition/expedition_worker_xla.py
```

If missing, add at the top of the color constants block:

```python
GOLD   = "\033[33m"
DIM    = "\033[2m"
RESET  = "\033[0m"
```

Then insert after `bestiary.save()`:

```python
        if ephemeral:
            _evicted, _freed = _janitor.maybe_evict(
                item.model_id, score, preexisting, evict_failures
            )
            if _evicted:
                _mb = _freed / 1_048_576
                print(f"    {DIM}♻ {_mb:.0f} MB freed{RESET}")
            elif success and _janitor.is_gold_star(score):
                print(f"    {GOLD}★ SAVED{RESET}")
```

- [ ] **Step 5: Add `--ephemeral` and `--evict-failures` to XLA arg parser**

Find the `parser.add_argument("--bench-shapes"` block in `expedition_worker_xla.py` (~line 1363) and add after it:

```python
    parser.add_argument("--ephemeral", action="store_true",
                        help="Evict net-new HF model weights after each result "
                             "unless the model earns a gold-star rating.")
    parser.add_argument("--evict-failures", action="store_true",
                        help="With --ephemeral, also evict weights for failed models.")
```

- [ ] **Step 6: Forward new args into `run_worker_xla()` call**

Find the `run_worker_xla(` call in `__main__` and add:

```python
        ephemeral=args.ephemeral,
        evict_failures=args.evict_failures,
```

- [ ] **Step 7: Verify**

```bash
python3 lib/expedition/expedition_worker_xla.py --help 2>&1 | grep -A2 "ephemeral\|evict"
```

Expected: same two flag descriptions as Task 2 Step 6.

- [ ] **Step 8: Run full test suite**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add lib/expedition/expedition_worker_xla.py
git commit -m "feat: wire --ephemeral/--evict-failures into XLA expedition worker"
```

---

## Task 4: Shell launcher env var forwarding

**Files:**
- Modify: `scripts/run_expedition.sh` — lines 60–98 (chip script generation block)

- [ ] **Step 1: Add env var reading at top of variable block**

Find the `NUM_CHIPS=4` / `RUN_NUMBER=1` block (~line 37) and add after the existing defaults:

```bash
EPHEMERAL=${EXPEDITION_EPHEMERAL:-0}
EVICT_FAILURES=${EXPEDITION_EVICT_FAILURES:-0}
```

- [ ] **Step 2: Conditionally append flags to the worker command**

Find the chip script generation block that ends with:

```bash
    --results /tmp/expedition_results_chip${chip_id}.csv
CHIPSCRIPT
```

Replace it with:

```bash
    --results /tmp/expedition_results_chip${chip_id}.csv \
$([ "${EXPEDITION_EPHEMERAL:-0}" = "1" ] && echo "    --ephemeral") \
$([ "${EXPEDITION_EVICT_FAILURES:-0}" = "1" ] && echo "    --evict-failures")
CHIPSCRIPT
```

- [ ] **Step 3: Verify the generated chip script has the flags when env is set**

```bash
EXPEDITION_EPHEMERAL=1 EXPEDITION_EVICT_FAILURES=1 \
  bash -c 'source scripts/run_expedition.sh --chips 1 --run 1 2>/dev/null; cat /tmp/expedition_chip_0.sh'
```

Expected: the chip script contains `--ephemeral` and `--evict-failures` in the `python3` command.

- [ ] **Step 4: Verify the flags are absent when env is not set**

```bash
bash -c 'source scripts/run_expedition.sh --chips 1 --run 1 2>/dev/null; cat /tmp/expedition_chip_0.sh'
```

Expected: the `python3` command has no `--ephemeral` or `--evict-failures`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_expedition.sh
git commit -m "feat: forward EXPEDITION_EPHEMERAL/EVICT_FAILURES env vars to chip workers"
```

---

## Task 5: `expedition.py` CLI flags

**Files:**
- Modify: `expedition.py` — arg parser (~line 1762), env block (~line 1937)

- [ ] **Step 1: Add `--ephemeral` and `--evict-failures` to the expedition CLI**

Find the `run_p.add_argument("--bench-shapes"` block (~line 1783) and add after it:

```python
    run_p.add_argument("--ephemeral", action="store_true",
                        help="Evict net-new HF model weights after each compile+bench "
                             "cycle. Gold-star models (rare/legendary or first-ever) "
                             "are always preserved.")
    run_p.add_argument("--evict-failures", action="store_true",
                        help="With --ephemeral, also evict weights for models that "
                             "fail to compile (default: keep failures for retry).")
```

- [ ] **Step 2: Set env vars before launching the shell script**

Find the `env = {**os.environ, ...}` block (~line 1937) and add two keys:

```python
    env = {**os.environ,
           "EXPEDITION_RUN":             str(run_number),
           "EXPEDITION_NUM_CHIPS":       str(num_chips),
           "EXPEDITION_BENCH_PASSES":    str(getattr(args, "bench_passes", 0)),
           "EXPEDITION_BENCH_SHAPES":    "1" if getattr(args, "bench_shapes", False) else "0",
           "EXPEDITION_EPHEMERAL":       "1" if getattr(args, "ephemeral", False) else "0",
           "EXPEDITION_EVICT_FAILURES":  "1" if getattr(args, "evict_failures", False) else "0",
           }
```

- [ ] **Step 3: Verify the help text**

```bash
python3 expedition.py run --help 2>&1 | grep -A3 "ephemeral\|evict"
```

Expected:
```
  --ephemeral           Evict net-new HF model weights after each compile+bench...
  --evict-failures      With --ephemeral, also evict weights for models that fail...
```

- [ ] **Step 4: Add a warning if `--evict-failures` is used without `--ephemeral`**

Find where `args = run_p.parse_args(...)` resolves (in the `run` subcommand handler) and add a guard. In `expedition.py`, the `run` subcommand handler starts around line 1720. Find the `def _cmd_run(args):` function and add at its top:

```python
    if getattr(args, "evict_failures", False) and not getattr(args, "ephemeral", False):
        print(f"  {_YELLOW}warning:{_RST} --evict-failures has no effect without --ephemeral")
```

- [ ] **Step 5: Run full test suite**

```bash
python3 -m pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add expedition.py
git commit -m "feat: add --ephemeral/--evict-failures CLI flags to expedition launcher"
```

---

## Self-Review

**Spec coverage:**
- ✅ `--ephemeral` flag: Task 2, 3, 4, 5
- ✅ `--evict-failures` flag: Task 2, 3, 4, 5
- ✅ Net-new-only eviction: `maybe_evict` checks `preexisting` before acting
- ✅ Gold star criteria (success + rare/legendary or first-ever): `is_gold_star()`
- ✅ `★ SAVED` TUI output: Task 2 Step 3, Task 3 Step 4
- ✅ `♻ X MB freed` TUI output: same
- ✅ Failures kept by default: `evict_failures=False` default in `maybe_evict`
- ✅ Static/local loaders silently skipped: `_hf_repo_dir` check in `maybe_evict`
- ✅ Error handling: `snapshot_preexisting` returns empty on error; `maybe_evict` catches rmtree errors
- ✅ Unit tests: Task 1 covers all decision-table rows
- ✅ Warning when `--evict-failures` without `--ephemeral`: Task 5 Step 4

**Placeholder scan:** None found.

**Type consistency:** `score` (ScoreResult), `preexisting` (frozenset[str]), `ephemeral`/`evict_failures` (bool) — consistent across all tasks. `_janitor` import alias used identically in Task 2 and Task 3.
