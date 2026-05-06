# Intelligent Dispatch + Multi-Chip Mesh + XLA First-Class — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make backend selection (forge vs xla) and chip count fully automatic per model, implement real multi-chip mesh dispatch with a "RALLY" game event, and close the three XLA first-class infrastructure gaps (bestiary backend field, Flax frontier discovery, pre-download pattern split).

**Architecture:** A new `router.py` module computes a `DispatchDecision(backend, chips, confidence, reason)` per model. The TUI becomes a real-time dispatcher — each chip is a slot, and models are dispatched one-at-a-time instead of giving each chip a full queue upfront. Mesh models wait in a holding slot until N chips free simultaneously, then a single multi-chip subprocess fires with a full-width RALLY banner replacing the chip grid. Three XLA infrastructure gaps are fixed in the same pass.

**Tech Stack:** Python 3.11, Textual (Textual 0.x), asyncio, JAX 0.7.1/pjrt-plugin-tt 0.9.0, tt-forge-fe, huggingface_hub, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `lib/expedition/bestiary.py` | Modify | Add `backend` param to `record_success()`; add `backends_succeeded` accumulator; add `is_compiled_by()` helper |
| `lib/expedition/scorer.py` | Modify | Replace flat `mesh_bonus` with `mesh_mult` multiplier; add `is_opportunist` and `is_formation_share` params |
| `lib/expedition/hf_discover.py` | Modify | Add `library: str | None = "pytorch"` to `discover_frontier()` and `discover_from_authors()` |
| `expedition.py` | Modify | Split `_IGNORE_PATTERNS` into two backend-specific lists; add `library` param to `_scan_frontier()` |
| `lib/expedition/router.py` | Create | `DispatchDecision` dataclass + `route_model()` function |
| `lib/expedition/expedition_worker.py` | Modify | Add `--model-json` single-model mode; switch CSV to append mode |
| `lib/expedition/expedition_worker_xla.py` | Modify | Add `--model-json` single-model mode; switch CSV to append mode |
| `expedition_tui.py` | Modify | SetupScreen auto mode; RunScreen per-model dispatcher; RallyBanner widget |
| `tests/expedition/test_intelligent_dispatch.py` | Create | Unit tests for all new logic (router, scorer, bestiary, dispatch) |

---

## Task 1: Bestiary backend field

**Files:**
- Modify: `lib/expedition/bestiary.py:174-229` (`record_success` method)
- Create: `tests/expedition/test_intelligent_dispatch.py` (first tests)

- [ ] **Step 1: Create test file with failing tests**

```python
# tests/expedition/test_intelligent_dispatch.py
"""Tests for intelligent dispatch: bestiary backend tracking, scorer mesh_mult,
router decisions, and worker single-model mode."""
import sys
from pathlib import Path

# Ensure project root is in sys.path when pytest runs from tests/ subdir
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import tempfile
import pytest
from lib.expedition.bestiary import Bestiary


# ── Task 1: Bestiary backend field ────────────────────────────────────────────

def _make_bestiary(tmp_path: Path) -> Bestiary:
    """Create a fresh in-memory Bestiary backed by a temp file."""
    b = Bestiary(path=str(tmp_path / "bestiary.json"))
    return b


def test_record_success_backend_stored(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success("org/model", chip=0, run=1, time_s=2.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="hello", backend="forge")
    assert b.compiled["org/model"]["backend"] == "forge"


def test_record_success_backends_succeeded_accumulates(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success("org/model", chip=0, run=1, time_s=2.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="hello", backend="forge")
    b.record_success("org/model", chip=1, run=2, time_s=1.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="world", backend="xla")
    assert b.compiled["org/model"]["backend"] == "forge"   # first backend preserved
    assert set(b.compiled["org/model"]["backends_succeeded"]) == {"forge", "xla"}


def test_is_compiled_by_returns_true_for_matching_backend(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success("org/model", chip=0, run=1, time_s=2.5, task="text-generation",
                     source="huggingface", rarity="common", hf_downloads=1000,
                     hf_created_at=None, artifact="hello", backend="xla")
    assert b.is_compiled_by("org/model", "xla") is True
    assert b.is_compiled_by("org/model", "forge") is False


def test_is_compiled_by_returns_false_for_unknown_model(tmp_path):
    b = _make_bestiary(tmp_path)
    assert b.is_compiled_by("nonexistent/model", "forge") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 -m pytest tests/expedition/test_intelligent_dispatch.py::test_record_success_backend_stored \
    tests/expedition/test_intelligent_dispatch.py::test_record_success_backends_succeeded_accumulates \
    tests/expedition/test_intelligent_dispatch.py::test_is_compiled_by_returns_true_for_matching_backend \
    tests/expedition/test_intelligent_dispatch.py::test_is_compiled_by_returns_false_for_unknown_model \
    -v 2>&1 | head -30
```

Expected: FAIL — `record_success() got unexpected keyword argument 'backend'`

- [ ] **Step 3: Add `backend` param to `record_success()` and add `is_compiled_by()`**

In `lib/expedition/bestiary.py`, change `record_success` signature and body:

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
    backend: str = "forge",      # NEW: "forge" or "xla"
) -> None:
    """Record a successful compilation.

    On first success, creates the compiled entry with all metadata. On
    subsequent calls, increments counters and updates best_time_s if the
    new run was faster. The artifact field is always overwritten with the
    most recent decoded inference output.

    Args:
        model_id:      HuggingFace model identifier (e.g. "openai/whisper-large-v3").
        chip:          Zero-based index of the Tenstorrent chip that ran this model.
        run:           Sequential run number within the current expedition session.
        time_s:        Wall-clock compilation + inference time in seconds.
        task:          HuggingFace pipeline task string (e.g. "automatic_speech_recognition").
        source:        Data origin: "huggingface", "local", etc.
        rarity:        Rarity tier from scorer.py: "common", "uncommon", "rare", "legendary".
        hf_downloads:  Monthly downloads from HuggingFace model card (None if unavailable).
        hf_created_at: ISO-8601 creation timestamp from HuggingFace (None if unavailable).
        artifact:      Decoded inference output string — the model's "voice" in the bestiary.
        backend:       Compilation frontend used: "forge" (PyTorch/ONNX) or "xla" (JAX/Flax).
    """
    now = datetime.now(timezone.utc).isoformat()
    if model_id not in self._data["compiled"]:
        # First-time entry: capture all immutable metadata from this run.
        self._data["compiled"][model_id] = {
            "first_compiled": now,
            "first_chip": chip,
            "run": run,
            "best_time_s": time_s,
            "successes": 0,
            "source": source,
            "task": task,
            "rarity": rarity,
            "hf_downloads": hf_downloads,
            "hf_created_at": hf_created_at,
            "artifact": artifact,
            "backend": backend,
            "backends_succeeded": [backend],
        }
    entry = self._data["compiled"][model_id]
    entry["successes"] += 1
    # Accumulate all backends that have successfully compiled this model.
    entry.setdefault("backends_succeeded", [entry.get("backend", "forge")])
    if backend not in entry["backends_succeeded"]:
        entry["backends_succeeded"].append(backend)
    # Track the fastest compilation time across all chips/runs.
    if time_s < entry["best_time_s"]:
        entry["best_time_s"] = time_s
    # Always update artifact so the bestiary reflects the most recent output.
    entry["artifact"] = artifact
```

Add `is_compiled_by()` after `is_compiled()` (around line 170):

```python
def is_compiled_by(self, model_id: str, backend: str) -> bool:
    """Return True if this model was successfully compiled by the given backend."""
    entry = self._data["compiled"].get(model_id)
    if not entry:
        return False
    return backend in entry.get("backends_succeeded", [entry.get("backend", "forge")])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_record_success or test_is_compiled" -v
```

Expected: 4 PASS

- [ ] **Step 5: Verify bestiary import still works**

```bash
python3 -c "from lib.expedition.bestiary import Bestiary; print('bestiary import OK')"
```

Expected: `bestiary import OK`

- [ ] **Step 6: Commit**

```bash
git add lib/expedition/bestiary.py tests/expedition/test_intelligent_dispatch.py
git commit -m "feat: add backend field to bestiary record_success + is_compiled_by helper"
```

---

## Task 2: Scorer mesh_mult + opportunist/formation_share bonuses

**Files:**
- Modify: `lib/expedition/scorer.py:106-179` (`compute_score` function)
- Modify: `tests/expedition/test_intelligent_dispatch.py` (add scorer tests)

- [ ] **Step 1: Add failing scorer tests**

Append to `tests/expedition/test_intelligent_dispatch.py`:

```python
# ── Task 2: Scorer mesh_mult ──────────────────────────────────────────────────

from lib.expedition.scorer import compute_score, Rarity, Newness


def test_mesh_mult_single_chip_unchanged():
    """1-chip compile should produce same points as before (mesh_mult = 1.0)."""
    s = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=1)
    assert s.pts == int(50 * 1.0 * 1.0 * 1.0 * 1.0)   # 50


def test_mesh_mult_four_chips():
    """4-chip compile earns 2.5× the single-chip score."""
    s1 = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=1)
    s4 = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=4)
    assert s4.pts == int(s1.pts * 2.5)


def test_mesh_mult_two_chips():
    """2-chip compile earns 1.5× the single-chip score."""
    s1 = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=1)
    s2 = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=2)
    assert s2.pts == int(s1.pts * 1.5)


def test_mesh_mult_in_breakdown():
    """Breakdown dict should expose mesh_mult and not have old mesh_bonus key."""
    s = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, mesh_chips=4)
    assert "mesh_mult" in s.breakdown
    assert "mesh_bonus" not in s.breakdown
    assert s.breakdown["mesh_mult"] == 2.5


def test_opportunist_bonus():
    """is_opportunist=True adds flat +25 after bracket."""
    s_plain = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0)
    s_opp   = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, is_opportunist=True)
    assert s_opp.pts == s_plain.pts + 25
    assert s_opp.breakdown["opportunist_bonus"] == 25


def test_formation_share_flat_150():
    """is_formation_share=True returns exactly 150 pts regardless of other params."""
    s = compute_score(True, True, Rarity.LEGENDARY, Newness.ZERO_DAY, 5, is_formation_share=True)
    assert s.pts == 150
    assert s.breakdown["formation_share"] is True


def test_legendary_4chip_rally_example():
    """Sanity-check the scoring example from the design spec."""
    # base=50, first_ever=100, first_voice=100 → 250
    # rarity=4.0, newness=5.0, streak=1.3, mesh=2.5
    # int(250 * 4.0 * 5.0 * 1.3 * 2.5) = 16250
    s = compute_score(
        success=True,
        is_first_ever=True,
        rarity=Rarity.LEGENDARY,
        newness=Newness.ZERO_DAY,
        streak=3,
        mesh_chips=4,
        is_first_voice=True,
    )
    assert s.pts == 16_250
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_mesh or test_opportunist or test_formation or test_legendary" -v 2>&1 | head -30
```

Expected: FAIL — `compute_score() got unexpected keyword argument 'is_opportunist'` and mesh tests fail on wrong pt values

- [ ] **Step 3: Rewrite `compute_score` in `lib/expedition/scorer.py`**

Replace the entire `compute_score` function (lines 106–179) with:

```python
def compute_score(
    success: bool,
    is_first_ever: bool,
    rarity: Rarity,
    newness: Newness,
    streak: int,
    mesh_chips: int = 1,
    is_first_voice: bool = False,
    is_opportunist: bool = False,
    is_formation_share: bool = False,
) -> ScoreResult:
    """Compute the expedition score for a single compile attempt.

    Formula (success path, normal single/mesh compile):
        mesh_mult = 1.0 + (mesh_chips - 1) * 0.5
        pts = int((base + first_ever_bonus + first_voice_bonus)
                  * rarity_mult * newness_mult * streak_mult * mesh_mult)
              + (25 if is_opportunist else 0)

    Special cases:
        is_formation_share=True → pts = 150 flat (non-lead mesh chip contribution)
        failure → pts = -10 flat

    Parameters
    ----------
    success:            Whether compilation succeeded.
    is_first_ever:      True if this is the first successful compile of this model ever.
    rarity:             Rarity tier (drives a multiplier on the combined base+bonus).
    newness:            How recently the model appeared on HF (multiplier, first-ever only).
    streak:             Consecutive successes before this one (+10% per, capped at 2x).
    mesh_chips:         Number of TT chips in the mesh. Drives mesh_mult multiplier.
    is_first_voice:     True when the model produced decoded meaningful output (+100 inside bracket).
    is_opportunist:     True when this model was compiled while a mesh was assembling (+25 flat after bracket).
    is_formation_share: True for non-lead chips in a mesh compile (returns 150 pts flat).

    Returns a ScoreResult with pts and a full breakdown dict for audit/display.
    """
    if not success:
        return ScoreResult(
            pts=-10, is_first_ever=is_first_ever, rarity=rarity, newness=newness,
            streak_at_score=streak, breakdown={"failure": -10},
        )

    # Non-lead mesh chip: flat contribution, bypasses normal formula entirely.
    if is_formation_share:
        return ScoreResult(
            pts=150, is_first_ever=False, rarity=rarity, newness=newness,
            streak_at_score=streak, breakdown={"formation_share": True},
        )

    base = 50
    first_ever_bonus  = 100 if is_first_ever else 0
    # First Voice: awarded when the compiled model produces real decoded output.
    first_voice_bonus = 100 if is_first_voice else 0
    rarity_mult  = _RARITY_MULT[rarity]
    newness_mult = _NEWNESS_MULT[newness] if is_first_ever else 1.0
    # Streak: +10% per consecutive success, hard-capped at 2x.
    streak_mult  = min(1.0 + streak * 0.1, 2.0)
    # Mesh multiplier: 1-chip = 1.0×, 2-chip = 1.5×, 4-chip = 2.5×.
    mesh_mult    = 1.0 + (mesh_chips - 1) * 0.5
    # Opportunist bonus: +25 flat when compiled while a mesh was assembling.
    opportunist_bonus = 25 if is_opportunist else 0

    pts = int(
        (base + first_ever_bonus + first_voice_bonus)
        * rarity_mult * newness_mult * streak_mult * mesh_mult
    ) + opportunist_bonus

    return ScoreResult(
        pts=pts,
        is_first_ever=is_first_ever,
        rarity=rarity,
        newness=newness,
        streak_at_score=streak,
        breakdown={
            "base":               base,
            "first_ever_bonus":   first_ever_bonus,
            "first_voice_bonus":  first_voice_bonus,
            "rarity_mult":        rarity_mult,
            "newness_mult":       newness_mult,
            "streak_mult":        streak_mult,
            "mesh_mult":          mesh_mult,
            "opportunist_bonus":  opportunist_bonus,
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_mesh or test_opportunist or test_formation or test_legendary" -v
```

Expected: 8 PASS

- [ ] **Step 5: Verify full test suite**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -v
```

Expected: All tests pass (no regressions on Task 1 tests)

- [ ] **Step 6: Verify syntax**

```bash
python3 -c "from lib.expedition.scorer import compute_score, Rarity, Newness; print('scorer import OK')"
```

Expected: `scorer import OK`

- [ ] **Step 7: Commit**

```bash
git add lib/expedition/scorer.py tests/expedition/test_intelligent_dispatch.py
git commit -m "feat: scorer mesh_mult multiplier + opportunist/formation_share bonuses"
```

---

## Task 3: HF discover `library` param

**Files:**
- Modify: `lib/expedition/hf_discover.py:225-293` (`discover_frontier`) and `:371-420` (`discover_from_authors`)
- Modify: `tests/expedition/test_intelligent_dispatch.py` (add hf_discover tests)

- [ ] **Step 1: Add failing tests for `discover_frontier` `library` param**

Append to `tests/expedition/test_intelligent_dispatch.py`:

```python
# ── Task 3: HF discover library param ────────────────────────────────────────

from unittest.mock import patch, MagicMock
from lib.expedition.hf_discover import discover_frontier, discover_from_authors


def test_discover_frontier_passes_library_to_api():
    """discover_frontier(library="jax") should call api.list_models(filter="jax", ...)."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_frontier(compiled_ids=set(), known_model_ids=set(), library="jax")
    call_kwargs = mock_api_instance.list_models.call_args[1]
    assert call_kwargs.get("filter") == "jax"


def test_discover_frontier_omits_filter_when_library_none():
    """discover_frontier(library=None) should NOT include 'filter' kwarg."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_frontier(compiled_ids=set(), known_model_ids=set(), library=None)
    call_kwargs = mock_api_instance.list_models.call_args[1]
    assert "filter" not in call_kwargs


def test_discover_frontier_default_library_is_pytorch():
    """Default call should still filter for pytorch (backwards compatible)."""
    mock_api_instance = MagicMock()
    mock_api_instance.list_models.return_value = []
    with patch("lib.expedition.hf_discover.HfApi", return_value=mock_api_instance):
        discover_frontier(compiled_ids=set(), known_model_ids=set())
    call_kwargs = mock_api_instance.list_models.call_args[1]
    assert call_kwargs.get("filter") == "pytorch"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_discover" -v 2>&1 | head -20
```

Expected: FAIL — `discover_frontier() got unexpected keyword argument 'library'`

- [ ] **Step 3: Add `library` param to `discover_frontier` in `lib/expedition/hf_discover.py`**

Change the function signature (line 225):

```python
def discover_frontier(
    compiled_ids: set[str],
    known_model_ids: set[str],
    limit: int = 1000,
    min_downloads: int = 0,
    min_likes: int = 0,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
    library: str | None = "pytorch",    # NEW: filter by HF library tag
) -> list[FrontierModel]:
```

Update the docstring's parameter list to include:
```
library:         HF library filter passed to api.list_models(). "pytorch" for forge
                 models, "jax" for Flax-native models, None for both (mixed mode).
```

Replace the `api.list_models(...)` call (lines 272–290) with:

```python
    api_kwargs = dict(
        sort="createdAt",
        direction=-1,   # descending → newest first
        limit=limit,
        expand=[
            "config",
            "pipeline_tag",
            "downloads",
            "likes",
            "gated",
            "disabled",
            "safetensors",
            "createdAt",
        ],
    )
    # Only pass filter when a library is specified — omitting it discovers all libraries.
    if library is not None:
        api_kwargs["filter"] = library
    try:
        hf_models = api.list_models(**api_kwargs)
    except Exception:
        return []
```

- [ ] **Step 4: Add `library` param to `discover_from_authors` in `lib/expedition/hf_discover.py`**

Find `discover_from_authors` (around line 371). Add `library: str | None = "pytorch"` to its signature:

```python
def discover_from_authors(
    authors: list[str],
    compiled_ids: set[str],
    known_model_ids: set[str],
    skip_gated: bool = True,
    library: str | None = "pytorch",    # NEW
) -> list[FrontierModel]:
```

Update the `api.list_models` call inside that function (around line 408) the same way:

```python
    for author in authors[:20]:
        try:
            author_kwargs = dict(
                author=author,
                sort="createdAt",
                direction=-1,
                limit=30,
                expand=["config", "pipeline_tag", "downloads", "likes", "gated",
                        "disabled", "safetensors", "createdAt"],
            )
            if library is not None:
                author_kwargs["filter"] = library
            author_models = list(api.list_models(**author_kwargs))
```

(The exact structure of this call may differ slightly — apply the same pattern of moving kwargs into a dict and conditionally adding `filter`.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_discover" -v
```

Expected: 3 PASS

- [ ] **Step 6: Verify hf_discover import**

```bash
python3 -c "from lib.expedition.hf_discover import discover_frontier, discover_from_authors; print('hf_discover import OK')"
```

Expected: `hf_discover import OK`

- [ ] **Step 7: Commit**

```bash
git add lib/expedition/hf_discover.py tests/expedition/test_intelligent_dispatch.py
git commit -m "feat: add library param to hf_discover — enables Flax-native frontier discovery"
```

---

## Task 4: expedition.py — pattern split + library threading

**Files:**
- Modify: `expedition.py:689-696` (`_IGNORE_PATTERNS` definition)
- Modify: `expedition.py:375-437` (`_scan_frontier` function)

- [ ] **Step 1: Add failing test for pattern split**

Append to `tests/expedition/test_intelligent_dispatch.py`:

```python
# ── Task 4: Pattern split ─────────────────────────────────────────────────────

def test_forge_ignore_patterns_excludes_msgpack():
    """Forge patterns must exclude .msgpack (Flax weights) since forge doesn't need them."""
    from expedition import _FORGE_IGNORE_PATTERNS
    assert any("msgpack" in p for p in _FORGE_IGNORE_PATTERNS)


def test_xla_ignore_patterns_includes_msgpack():
    """XLA patterns must NOT exclude .msgpack (Flax weights are required by XLA)."""
    from expedition import _XLA_IGNORE_PATTERNS
    assert not any("msgpack" in p for p in _XLA_IGNORE_PATTERNS)


def test_scan_frontier_accepts_library_param():
    """_scan_frontier must accept a library keyword argument without raising."""
    from expedition import _scan_frontier
    import inspect
    sig = inspect.signature(_scan_frontier)
    assert "library" in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_forge_ignore or test_xla_ignore or test_scan_frontier_accepts" -v 2>&1 | head -20
```

Expected: FAIL — `cannot import name '_FORGE_IGNORE_PATTERNS'` and `_scan_frontier()` has no `library` param

- [ ] **Step 3: Replace `_IGNORE_PATTERNS` with two backend-specific lists in `expedition.py`**

Find lines 685–696 in `expedition.py` and replace:

```python
# File patterns to skip when mirroring a model from HuggingFace.
# These are large binary formats specific to Flax/TF/Rust/OpenNMT that
# tt-forge never consumes; excluding them saves significant download time
# and disk space for multi-billion-parameter models.
_IGNORE_PATTERNS = [
    "*.msgpack",   # Flax/JAX checkpoints
    "*.h5",        # Keras/TF HDF5 weights
    "flax_model*", # Flax model shards
    "tf_model*",   # TensorFlow SavedModel
    "rust_model*", # Rust/candle weights
    "*.ot",        # OpenNMT tokenizer files
]
```

With:

```python
# Patterns to ignore when pre-downloading for forge-onnx backend.
# Forge only needs PyTorch safetensors — skip Flax/TF/Keras formats.
_FORGE_IGNORE_PATTERNS = [
    "*.msgpack",   # Flax/JAX checkpoints (not needed by forge)
    "flax_model*", # Flax model shards
    "*.h5",        # Keras/TF HDF5 weights
    "tf_model*",   # TensorFlow SavedModel
    "rust_model*", # Rust/candle weights
    "*.ot",        # OpenNMT tokenizer files
]

# Patterns to ignore when pre-downloading for tt-xla backend.
# XLA needs Flax weights (.msgpack) — skip TF/Keras/Rust/PyTorch-only formats.
_XLA_IGNORE_PATTERNS = [
    "*.h5",        # Keras/TF HDF5 weights
    "tf_model*",   # TensorFlow SavedModel
    "rust_model*", # Rust/candle weights
    "*.ot",        # OpenNMT tokenizer files
]

# Default patterns for CLI paths that don't route per-model (conservative: forge-safe).
_IGNORE_PATTERNS = _FORGE_IGNORE_PATTERNS
```

This preserves the `_IGNORE_PATTERNS` name for any code paths in `expedition.py` that reference it (the CLI pre-download path uses it at lines 832 and 874 — they'll now use forge patterns, which is correct for the non-TUI CLI).

- [ ] **Step 4: Add `library` param to `_scan_frontier` in `expedition.py`**

Find `_scan_frontier` (line 375) and update its signature:

```python
def _scan_frontier(
    bestiary_compiled_ids: set[str],
    forge_model_ids: set[str],
    min_downloads: int = 0,
    min_likes: int = 0,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
    proven_authors: set[str] | None = None,
    library: str | None = "pytorch",        # NEW
) -> list[dict]:
```

Inside the function body, thread `library` through both calls:

```python
    models = discover_frontier(
        compiled_ids=bestiary_compiled_ids,
        known_model_ids=forge_model_ids,
        min_downloads=min_downloads,
        min_likes=min_likes,
        max_params_b=max_params_b,
        skip_gated=skip_gated,
        library=library,                     # NEW
    )
    ...
    if proven_authors and len(models) < 8:
        supplement = discover_from_authors(
            authors=list(proven_authors),
            compiled_ids=bestiary_compiled_ids,
            known_model_ids=forge_model_ids,
            skip_gated=skip_gated,
            library=library,                 # NEW
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_forge_ignore or test_xla_ignore or test_scan_frontier_accepts" -v
```

Expected: 3 PASS

- [ ] **Step 6: Verify expedition.py syntax**

```bash
python3 -c "import py_compile; py_compile.compile('expedition.py', doraise=True); print('expedition.py syntax OK')"
```

Expected: `expedition.py syntax OK`

- [ ] **Step 7: Commit**

```bash
git add expedition.py tests/expedition/test_intelligent_dispatch.py
git commit -m "feat: split _IGNORE_PATTERNS by backend; thread library param through _scan_frontier"
```

---

## Task 5: `lib/expedition/router.py` (new file)

**Files:**
- Create: `lib/expedition/router.py`
- Modify: `tests/expedition/test_intelligent_dispatch.py` (add router tests)

- [ ] **Step 1: Add failing router tests**

Append to `tests/expedition/test_intelligent_dispatch.py`:

```python
# ── Task 5: Router ───────────────────────────────────────────────────────────

def test_router_jax_native_routes_to_xla(tmp_path):
    """Models with library=="jax" or "flax" must be routed to XLA backend."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "google/flax-bert", "library": "jax",
         "hf_downloads": 5000, "mesh_chips": 1},
        b,
    )
    assert d.backend == "xla"
    assert d.confidence >= 0.9
    assert d.reason == "jax-native"


def test_router_flax_library_routes_to_xla(tmp_path):
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "google/flax-bert", "library": "flax",
         "hf_downloads": 5000, "mesh_chips": 1},
        b,
    )
    assert d.backend == "xla"
    assert d.reason == "jax-native"


def test_router_default_routes_to_forge(tmp_path):
    """Pytorch models with no special signals should default to forge."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "foo/bar", "library": "pytorch",
         "hf_downloads": 100, "mesh_chips": 1},
        b,
    )
    assert d.backend == "forge"
    assert d.reason == "default"


def test_router_forge_failure_history_redirects_to_xla(tmp_path):
    """Models with ≥2 forge_missing_op or forge_internal failures redirect to XLA."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    # Simulate 2 forge_missing_op failures recorded in bestiary
    b._data["failed"]["org/troubled"] = {
        "run_first_failed": 1, "attempts": 2,
        "last_error": "are not implemented in tt-forge",
        "error_category": "forge_missing_op",
    }
    d = route_model(
        {"model_id": "org/troubled", "library": "pytorch",
         "hf_downloads": 1000, "mesh_chips": 1},
        b,
    )
    assert d.backend == "xla"
    assert d.reason == "forge-failure-history"


def test_router_large_model_gets_4_chips(tmp_path):
    """Models with mesh_chips=4 in metadata should be dispatched on 4 chips."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "deepseek-ai/deepseek-v3", "library": "pytorch",
         "hf_downloads": 1_000_000, "mesh_chips": 4, "hf_params_b": 67.0},
        b,
        available_chips=set(range(4)),
    )
    assert d.chips == 4


def test_router_caps_chips_at_available(tmp_path):
    """If only 2 chips available but model wants 4, cap at 2."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model(
        {"model_id": "deepseek-ai/deepseek-v3", "library": "pytorch",
         "hf_downloads": 1_000_000, "mesh_chips": 4},
        b,
        available_chips=set(range(2)),
    )
    assert d.chips == 2


def test_dispatch_decision_has_required_fields(tmp_path):
    """DispatchDecision must have backend, chips, confidence, reason."""
    from lib.expedition.router import route_model
    b = _make_bestiary(tmp_path)
    d = route_model({"model_id": "a/b", "library": "pytorch",
                     "hf_downloads": 100, "mesh_chips": 1}, b)
    assert hasattr(d, "backend")
    assert hasattr(d, "chips")
    assert hasattr(d, "confidence")
    assert hasattr(d, "reason")
    assert isinstance(d.chips, int)
    assert 0.0 <= d.confidence <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_router or test_dispatch_decision" -v 2>&1 | head -20
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lib.expedition.router'`

- [ ] **Step 3: Create `lib/expedition/router.py`**

```python
# lib/expedition/router.py
"""Per-model routing decisions: which backend (forge/xla) and how many chips.

This module is stateless — it inspects a queue item dict and a Bestiary snapshot
and returns a DispatchDecision.  It has no side effects and imports nothing from
the TUI or worker layers.
"""
from __future__ import annotations

from dataclasses import dataclass

from lib.expedition.bestiary import Bestiary

# Architectures whose canonical implementation is Flax-native and that have
# already proven out on tt-xla.  Models reporting these model_type values
# are routed to the XLA backend at moderate confidence.
_XLA_AFFINITY_TYPES: frozenset[str] = frozenset({
    "flax_bert",
    "flax_gpt2",
    "flax_roberta",
    "flax_t5",
})

# Error categories that indicate forge cannot handle a model.  Two or more
# failures in these categories redirect future attempts to the XLA backend.
_FORGE_FATAL_CATEGORIES: frozenset[str] = frozenset({
    "forge_missing_op",
    "forge_internal",
})


@dataclass
class DispatchDecision:
    """Routing decision for a single model."""
    backend: str       # "forge" or "xla"
    chips: int         # 1, 2, or 4
    confidence: float  # 0.0–1.0, informational for UI display
    reason: str        # short label: "jax-native", "forge-failure-history", etc.


def route_model(
    item: dict,
    bestiary: Bestiary,
    available_chips: set[int] | None = None,
) -> DispatchDecision:
    """Compute a DispatchDecision for a single queue item.

    Priority order (first match wins):
      1. library == "jax" or "flax" → xla, confidence=0.92, reason="jax-native"
      2. ≥2 forge fatal failures in bestiary → xla, confidence=0.75, reason="forge-failure-history"
      3. model_type in _XLA_AFFINITY_TYPES → xla, confidence=0.68, reason="arch-xla-affinity"
      4. default → forge, confidence=0.60, reason="default"

    Chip count:
      - item["mesh_chips"] is the primary source (set by hf_discover.py heuristics).
      - Falls back to 1 chip if not present.
      - Capped at len(available_chips) if provided.

    Args:
        item:            Queue item dict (model_id, library, mesh_chips, etc.).
        bestiary:        Loaded Bestiary for failure history lookups.
        available_chips: Set of chip IDs in this run (used for cap only).
                         Pass the full chip set (not just free chips) — the TUI
                         enforces free-chip quorum separately.
    """
    model_id   = item.get("model_id", "")
    library    = (item.get("library") or "").lower()
    model_type = (item.get("model_type") or "").lower()

    # ── Backend routing ───────────────────────────────────────────────────────

    # Priority 1: JAX/Flax library tag is definitive.
    if library in ("jax", "flax"):
        backend    = "xla"
        confidence = 0.92
        reason     = "jax-native"

    # Priority 2: Forge has already failed this model with a fundamental error.
    elif _has_forge_fatal_history(model_id, bestiary):
        backend    = "xla"
        confidence = 0.75
        reason     = "forge-failure-history"

    # Priority 3: Architecture is known to work well on XLA.
    elif model_type in _XLA_AFFINITY_TYPES:
        backend    = "xla"
        confidence = 0.68
        reason     = "arch-xla-affinity"

    # Priority 4: Default to forge.
    else:
        backend    = "forge"
        confidence = 0.60
        reason     = "default"

    # ── Chip count ────────────────────────────────────────────────────────────
    chips = int(item.get("mesh_chips", 1)) or 1
    if available_chips is not None and chips > len(available_chips):
        chips = max(1, len(available_chips))

    return DispatchDecision(backend=backend, chips=chips,
                            confidence=confidence, reason=reason)


def _has_forge_fatal_history(model_id: str, bestiary: Bestiary) -> bool:
    """Return True if this model has ≥2 forge-fatal failures recorded."""
    entry = bestiary.failed.get(model_id)
    if not entry:
        return False
    if entry.get("error_category") not in _FORGE_FATAL_CATEGORIES:
        return False
    return int(entry.get("attempts", 0)) >= 2
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_router or test_dispatch_decision" -v
```

Expected: 8 PASS

- [ ] **Step 5: Verify router import**

```bash
python3 -c "from lib.expedition.router import route_model, DispatchDecision; print('router import OK')"
```

Expected: `router import OK`

- [ ] **Step 6: Commit**

```bash
git add lib/expedition/router.py tests/expedition/test_intelligent_dispatch.py
git commit -m "feat: add router.py — DispatchDecision + route_model() per-model dispatch decisions"
```

---

## Task 6: Worker single-model mode + CSV append

**Files:**
- Modify: `lib/expedition/expedition_worker.py:469-480` (queue loading), `:757` (CSV open mode), `:778-800` (argparse)
- Modify: `lib/expedition/expedition_worker_xla.py:445-448` (queue loading), `:750` (CSV open mode), `:766-784` (argparse)
- Modify: `tests/expedition/test_intelligent_dispatch.py` (add worker tests)

- [ ] **Step 1: Add failing worker tests**

Append to `tests/expedition/test_intelligent_dispatch.py`:

```python
# ── Task 6: Worker single-model mode ──────────────────────────────────────────

import subprocess


def test_forge_worker_accepts_model_json_flag():
    """expedition_worker.py --help should list --model-json as an argument."""
    result = subprocess.run(
        ["python3", "lib/expedition/expedition_worker.py", "--help"],
        capture_output=True, text=True,
        cwd="/home/ttuser/code/tt-forge-compiletron",
    )
    assert "--model-json" in result.stdout


def test_xla_worker_accepts_model_json_flag():
    """expedition_worker_xla.py --help should list --model-json as an argument."""
    result = subprocess.run(
        ["python3", "lib/expedition/expedition_worker_xla.py", "--help"],
        capture_output=True, text=True,
        cwd="/home/ttuser/code/tt-forge-compiletron",
    )
    assert "--model-json" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_forge_worker or test_xla_worker" -v 2>&1 | head -20
```

Expected: FAIL — `--model-json` not in help output

- [ ] **Step 3: Add `--model-json` to `expedition_worker.py` argparse and queue loading**

At the end of `expedition_worker.py` (lines 778–800), add `--model-json` to argparse:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-chip Expedition worker: compile, decode, and score a model queue."
    )
    parser.add_argument("--chip",       type=int, required=True,
                        help="Zero-based index of the TT chip this worker owns.")
    parser.add_argument("--run",        type=int, required=True,
                        help="Sequential expedition run number.")
    parser.add_argument("--bestiary",   default="data/bestiary.json",
                        help="Path to the bestiary JSON file.")
    parser.add_argument("--queue",      default=None,
                        help="Path to this chip's queue JSON file.")
    parser.add_argument("--model-json", default=None,
                        help="Path to a single-model JSON file. Overrides --queue.")
    parser.add_argument("--results",    required=True,
                        help="Path to write the per-chip CSV results file.")
    args = parser.parse_args()
    if not args.queue and not args.model_json:
        parser.error("one of --queue or --model-json is required")

    run_worker(
        chip_id=args.chip,
        run_number=args.run,
        bestiary_path=args.bestiary,
        queue_path=args.queue,
        model_json_path=args.model_json,
        results_path=args.results,
    )
```

Update `run_worker` signature and queue-loading logic in `expedition_worker.py`. Find `_load_queue` (line 469) and `run_worker` (line 543):

Change `run_worker` signature:
```python
def run_worker(chip_id: int, run_number: int, bestiary_path: str,
               queue_path: str | None, results_path: str,
               model_json_path: str | None = None) -> None:
```

Change queue loading inside `run_worker` (find the line `queue = _load_queue(queue_path)` around line 571):
```python
    if model_json_path:
        queue = [_load_single_model(model_json_path)]
    elif queue_path:
        queue = _load_queue(queue_path)
    else:
        raise ValueError("Either queue_path or model_json_path must be provided")
```

Add `_load_single_model` helper function next to `_load_queue`:
```python
def _load_single_model(model_json_path: str) -> QueueItem:
    """Load a single model JSON written by the TUI dispatcher."""
    with open(model_json_path) as f:
        data = json.load(f)
    return QueueItem(**data)
```

Change CSV open mode from `"w"` to `"a"` (line 757) — append mode so per-model calls accumulate into the results file:
```python
    with open(results_path, "a", newline="") as f:
```

Also change the header write to only write it when the file is empty:
```python
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_file_empty = not Path(results_path).exists() or Path(results_path).stat().st_size == 0
    with open(results_path, "a", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
            if results_file_empty:
                writer.writeheader()
            writer.writerows(results)
```

- [ ] **Step 4: Apply same changes to `expedition_worker_xla.py`**

Update argparse at lines 766–784:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-chip XLA Expedition worker: JAX/Flax compile, decode, and score."
    )
    parser.add_argument("--chip",       type=int, required=True)
    parser.add_argument("--run",        type=int, required=True)
    parser.add_argument("--bestiary",   default="data/bestiary.json")
    parser.add_argument("--queue",      default=None)
    parser.add_argument("--model-json", default=None,
                        help="Path to a single-model JSON file. Overrides --queue.")
    parser.add_argument("--results",    required=True)
    args = parser.parse_args()
    if not args.queue and not args.model_json:
        parser.error("one of --queue or --model-json is required")

    run_worker_xla(
        chip_id=args.chip,
        run_number=args.run,
        bestiary_path=args.bestiary,
        queue_path=args.queue,
        model_json_path=args.model_json,
        results_path=args.results,
    )
```

Find `run_worker_xla` signature and update it:
```python
def run_worker_xla(chip_id: int, run_number: int, bestiary_path: str,
                   queue_path: str | None, results_path: str,
                   model_json_path: str | None = None) -> None:
```

Find `_load_queue` at line 445 in the XLA worker:
```python
def _load_queue(queue_path: str) -> list[QueueItem]:
    with open(queue_path) as f:
        items = json.load(f)
    return [QueueItem(**item) for item in items]


def _load_single_model_xla(model_json_path: str) -> QueueItem:
    """Load a single model JSON written by the TUI dispatcher."""
    with open(model_json_path) as f:
        data = json.load(f)
    return QueueItem(**data)
```

Update queue loading inside `run_worker_xla` (find `queue = _load_queue(queue_path)`):
```python
    if model_json_path:
        queue = [_load_single_model_xla(model_json_path)]
    elif queue_path:
        queue = _load_queue(queue_path)
    else:
        raise ValueError("Either queue_path or model_json_path must be provided")
```

Change CSV open mode at line 750 to append with header guard:
```python
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    results_file_empty = not Path(results_path).exists() or Path(results_path).stat().st_size == 0
    with open(results_path, "a", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
            if results_file_empty:
                writer.writeheader()
            writer.writerows(results)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -k "test_forge_worker or test_xla_worker" -v
```

Expected: 2 PASS

- [ ] **Step 6: Verify syntax on both workers**

```bash
python3 -c "
import py_compile
for f in ['lib/expedition/expedition_worker.py', 'lib/expedition/expedition_worker_xla.py']:
    py_compile.compile(f, doraise=True)
    print('ok', f)
"
```

Expected: `ok lib/expedition/expedition_worker.py` and `ok lib/expedition/expedition_worker_xla.py`

- [ ] **Step 7: Commit**

```bash
git add lib/expedition/expedition_worker.py lib/expedition/expedition_worker_xla.py tests/expedition/test_intelligent_dispatch.py
git commit -m "feat: workers accept --model-json for per-model TUI dispatch + CSV append mode"
```

---

## Task 7: SetupScreen `auto` backend mode

**Files:**
- Modify: `expedition_tui.py:375` (`self._backend` init), `:428` (backend display string), `:500` (`action_cycle_backend`), `:570-571` (`fw_map`), `:588-595` (`_scan_frontier` call)

This task has no unit test (TUI widget state is hard to test in isolation without running Textual). Verification is a dry-run import + syntax check.

- [ ] **Step 1: Add `"auto"` to the backend cycle in `action_cycle_backend`**

Find line 500 in `expedition_tui.py`:
```python
def action_cycle_backend(self)   -> None: self._backend = {"forge": "xla", "xla": "mixed", "mixed": "forge"}[self._backend]
```

Replace with:
```python
def action_cycle_backend(self) -> None:
    self._backend = {"auto": "forge", "forge": "xla", "xla": "mixed", "mixed": "auto"}[self._backend]
```

- [ ] **Step 2: Set default backend to `"auto"` and update display string**

Find line 375:
```python
        self._backend      = "forge"  # forge | xla | mixed
```

Change to:
```python
        self._backend      = "auto"   # auto | forge | xla | mixed
```

Find line 394:
```python
        self._backend              = getattr(app, "backend", "forge")
```

Change to:
```python
        self._backend              = getattr(app, "backend", "auto")
```

Find line 428 (backend display string dict):
```python
        backend_str = {"forge": "[bold]forge[/]", "xla": "[bold cyan]XLA[/]", "mixed": "[bold yellow]MIXED[/]"}.get(self._backend, self._backend)
```

Change to:
```python
        backend_str = {
            "auto":  "[bold green]AUTO[/]  [dim]routes per-model[/]",
            "forge": "[bold]forge[/]",
            "xla":   "[bold cyan]XLA[/]",
            "mixed": "[bold yellow]MIXED[/]",
        }.get(self._backend, self._backend)
```

- [ ] **Step 3: Add `"auto"` to `fw_map` and thread `library` into `_scan_frontier` call**

Find lines 570–571:
```python
            fw_map = {"forge": "pytorch", "xla": "jax", "mixed": None}
            scan_fw = fw_map.get(self._backend, "pytorch")
```

Change to:
```python
            fw_map = {"auto": None, "forge": "pytorch", "xla": "jax", "mixed": None}
            scan_fw = fw_map.get(self._backend, "pytorch")
```

Find the `_scan_frontier(...)` call around lines 588–595:
```python
            frontier_items = _scan_frontier(
                compiled_ids,
                forge_ids,
                min_downloads = self._min_downloads,
                min_likes     = self._min_likes,
                max_params_b  = self._max_params_b,
                skip_gated    = not self._allow_gated,
            )
```

Change to:
```python
            frontier_items = _scan_frontier(
                compiled_ids,
                forge_ids,
                min_downloads = self._min_downloads,
                min_likes     = self._min_likes,
                max_params_b  = self._max_params_b,
                skip_gated    = not self._allow_gated,
                library       = scan_fw,
            )
```

- [ ] **Step 4: Update `ExpeditionTUI.backend` default and pass through to `RunScreen`**

Find `ExpeditionTUI` class definition. Find where `app.backend` is set or where it defaults. Around line 394 in `SetupScreen.on_mount`:
```python
        self._backend = getattr(app, "backend", "forge")
```

Find `ExpeditionTUI.__init__` or its class-level attribute. Search for `backend` in the app class and update any default from `"forge"` to `"auto"`.

Also find where `RunScreen` is constructed (around line 691):
```python
                backend      = self._backend,
```

Verify this line already passes `self._backend` — it does, so no change needed there.

- [ ] **Step 5: Verify syntax and import**

```bash
python3 -c "import py_compile; py_compile.compile('expedition_tui.py', doraise=True); print('expedition_tui.py syntax OK')"
python3 -c "from expedition_tui import ExpeditionTUI; print('TUI import OK')"
```

Expected: Both print `OK`

- [ ] **Step 6: Commit**

```bash
git add expedition_tui.py
git commit -m "feat: SetupScreen adds auto backend mode — routes per model via router.py"
```

---

## Task 8: RunScreen per-model dispatcher

**Files:**
- Modify: `expedition_tui.py:752-887` (RunScreen `__init__`, `on_mount`, `_launch_chip`)

This is the largest single change. The `_launch_chip` worker is replaced by a dispatcher loop. Verification is a syntax check + TUI import; end-to-end dispatch requires hardware.

- [ ] **Step 1: Add `_model_pool` and dispatcher state to `RunScreen.__init__`**

Find `RunScreen.__init__` (line 752) and extend it:

```python
    def __init__(self, chip_queues: list[list[dict]], num_chips: int,
                 run_number: int, arch: str, project_dir: Path,
                 backend: str = "auto", **kwargs) -> None:
        super().__init__(**kwargs)
        self.chip_queues  = chip_queues
        self.num_chips    = num_chips
        self.run_number   = run_number
        self.arch         = arch
        self._project_dir = project_dir
        self.backend      = backend   # auto | forge | xla | mixed
        self._chip_rarity: list[str] = ["common"] * 4
        self._chip_streak: list[int] = [0] * 4
        self._chip_best:   list[int] = [0] * 4
        self._done_count  = 0
        # ── Per-model dispatcher state ────────────────────────────────────────
        # Flatten chip_queues round-robin into a single ordered pool.
        self._model_pool: list[dict] = []
        for i in range(max(len(q) for q in chip_queues) if chip_queues else 0):
            for q in chip_queues:
                if i < len(q):
                    self._model_pool.append(q[i])
        self._free_chips: set[int]  = set(range(num_chips))
        self._mesh_holding: dict | None = None       # model waiting for chip quorum
        self._opportunist_active: bool  = False
        self._chip_first_dispatch: set[int] = set()  # tracks stagger state
        self._bestiary = None                         # loaded at on_mount
```

- [ ] **Step 2: Update `on_mount` to load bestiary and seed dispatcher**

Find `on_mount` (line 792) and replace the loop at the end:

```python
    def on_mount(self) -> None:
        # Write a placeholder run JSON so the run counter advances correctly.
        try:
            runs_dir = self._project_dir / "data" / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_file = runs_dir / f"run_{self.run_number:03d}.json"
            if not run_file.exists():
                run_file.write_text(json.dumps({
                    "run":       self.run_number,
                    "timestamp": __import__("datetime").datetime.now().isoformat(),
                    "chips":     self.num_chips,
                    "tui":       True,
                }, indent=2))
        except Exception:
            pass

        # Load bestiary for router queries (read-only at run time).
        from lib.expedition.bestiary import Bestiary as _Bestiary
        self._bestiary = _Bestiary(path=str(self._project_dir / "data" / "bestiary.json"))

        # Seed the dispatcher — each free chip gets its first model.
        self._dispatch_next()
```

- [ ] **Step 3: Add `_get_decision()` helper**

After `on_mount`, add:

```python
    def _get_decision(self, model: dict):
        """Compute a DispatchDecision for model, respecting self.backend override."""
        from lib.expedition.router import route_model, DispatchDecision
        if self.backend == "auto":
            return route_model(model, self._bestiary,
                               available_chips=set(range(self.num_chips)))
        else:
            # Manual override: honour user's backend choice, use model's mesh_chips.
            return DispatchDecision(
                backend=self.backend if self.backend != "mixed" else _chip_backend(0, "mixed"),
                chips=model.get("mesh_chips", 1),
                confidence=1.0,
                reason="manual",
            )
```

- [ ] **Step 4: Add `_dispatch_next()` method**

```python
    def _dispatch_next(self) -> None:
        """Find the next dispatchable model and launch it. Called at mount and after each chip completes."""
        # Check if a waiting mesh model now has quorum.
        if self._mesh_holding:
            chips_needed: int = self._mesh_holding["chips_needed"]
            if len(self._free_chips) >= chips_needed:
                chip_ids = sorted(self._free_chips)[:chips_needed]
                self._fire_rally(self._mesh_holding, chip_ids)
                return

        # Scan the pool for a dispatchable model.
        for i, model in enumerate(self._model_pool):
            decision = self._get_decision(model)

            if decision.chips == 1:
                if self._free_chips:
                    chip_id = min(self._free_chips)
                    self._free_chips.discard(chip_id)
                    self._model_pool.pop(i)
                    self._launch_model(chip_id, model, decision)
                    # Keep scanning — other free chips may still need work.
                    self._dispatch_next()
                    return
            else:
                # Multi-chip model: hold it and keep scanning for single-chip work.
                if self._mesh_holding is None:
                    self._mesh_holding = {
                        **model,
                        "chips_needed": decision.chips,
                        "decision": decision,
                    }
                    self._opportunist_active = True
                    self._model_pool.pop(i)
                    try:
                        el = self.query_one("#event-log", EventLog)
                        el.write(
                            f"[yellow]⏳ MESH ASSEMBLING — "
                            f"{model.get('model_id', '?').split('/')[-1]} "
                            f"needs {decision.chips} chips[/]"
                        )
                    except Exception:
                        pass
                    # Keep scanning pool for single-chip models.
                    self._dispatch_next()
                    return
                # Another mesh model already holding — skip for now.
                continue

        # Check if run is complete: pool empty, no mesh holding, all chips free.
        if not self._model_pool and not self._mesh_holding and len(self._free_chips) == self.num_chips:
            self._on_all_done()
```

- [ ] **Step 5: Add `_launch_model()` method**

```python
    @work
    async def _launch_model(self, chip_id: int, model: dict, decision, mesh_chip_ids: list[int] | None = None) -> None:
        """Launch a worker subprocess for one model on one or more chips."""
        import tempfile

        # First dispatch to each chip: stagger by chip_id * 2 seconds.
        if chip_id not in self._chip_first_dispatch:
            self._chip_first_dispatch.add(chip_id)
            if chip_id > 0:
                await asyncio.sleep(chip_id * 2)

        # Write the model dict to a temp JSON file.
        model_json_path = f"/tmp/expedition_model_chip{chip_id}.json"
        Path(model_json_path).write_text(json.dumps(model))

        # Results CSV path (append mode — one file per chip across all models).
        results_path = f"/tmp/expedition_results_chip{chip_id}.csv"

        # Determine backend (auto: from decision; mixed: per chip; manual: from decision).
        if self.backend == "mixed":
            chip_be = _chip_backend(chip_id, "mixed")
        else:
            chip_be = decision.backend

        # Build env for this chip.
        if mesh_chip_ids:
            visible = ",".join(str(c) for c in mesh_chip_ids)
        else:
            visible = str(chip_id)

        if chip_be == "xla":
            python_exe  = str(self._project_dir / "xla-venv" / "bin" / "python3")
            worker_path = str(self._project_dir / "lib" / "expedition" / "expedition_worker_xla.py")
            env = {k: v for k, v in os.environ.items() if k != "TT_METAL_HOME"}
            env.update({
                "TT_VISIBLE_DEVICES":    visible,
                "TT_METAL_LOGGER_LEVEL": "FATAL",
                "JAX_PLATFORMS":         "tt",
                "PYTHONUNBUFFERED":      "1",
            })
        else:
            python_exe  = sys.executable
            worker_path = str(self._project_dir / "lib" / "expedition" / "expedition_worker.py")
            env = {
                **os.environ,
                "TT_VISIBLE_DEVICES":      visible,
                "TT_METAL_ARCH_NAME":      self.arch,
                "TT_METAL_LOGGER_LEVEL":   "FATAL",
                "TT_MESH_GRAPH_DESC_PATH": str(
                    self._project_dir / "mesh_graph_descriptors"
                    / "p100_mesh_graph_descriptor.textproto"
                ),
                "PYTHONUNBUFFERED":        "1",
            }

        # Write confidence label to the chip panel.
        try:
            panel = self.query_one(f"#chip-{chip_id}", ChipPanel)
            chip_label = mesh_chip_ids or [chip_id]
            panel.write_line(
                f"\033[2m  routing: {chip_be} · conf {decision.confidence:.2f} "
                f"· {len(chip_label)}-chip\033[0m\n"
            )
        except Exception:
            pass

        proc = await asyncio.create_subprocess_exec(
            python_exe,
            worker_path,
            "--chip",       str(chip_id),
            "--run",        str(self.run_number),
            "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
            "--model-json", model_json_path,
            "--results",    results_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )

        try:
            panel = self.query_one(f"#chip-{chip_id}", ChipPanel)
        except Exception:
            panel = None

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            if panel:
                panel.write_line(line)
            self._parse_for_events(chip_id, line)

        await proc.wait()
        if panel:
            panel.mark_done(proc.returncode == 0)

        status = _read_status(chip_id)
        pts    = int(status.get("pts", 0))
        try:
            el = self.query_one("#event-log", EventLog)
            el.log_chip_done(chip_id, pts, self._chip_best[chip_id])
        except Exception:
            pass

        # Free the chip(s) and continue dispatching.
        if mesh_chip_ids:
            for cid in mesh_chip_ids:
                self._on_chip_free(cid)
            # Hide RALLY banner, restore chip grid.
            try:
                self.query_one("#rally-banner").display = False
                self.query_one("#chip-grid").display    = True
            except Exception:
                pass
        else:
            self._on_chip_free(chip_id)
```

- [ ] **Step 6: Add `_on_chip_free()` and `_on_all_done()` methods**

```python
    def _on_chip_free(self, chip_id: int) -> None:
        self._free_chips.add(chip_id)
        self._done_count += 1
        self._dispatch_next()

    @work
    async def _on_all_done(self) -> None:
        try:
            el = self.query_one("#event-log", EventLog)
            el.write(f"\n[bold green]{'═'*34}[/]")
            el.write("[bold green]  ⚡ ALL CHIPS COMPLETE[/]")
            for n in (3, 2, 1):
                el.write(f"[dim]  → Results in {n}...[/]")
                await asyncio.sleep(0.8)
            el.write(f"[bold green]{'═'*34}[/]")
        except Exception:
            await asyncio.sleep(2.4)
        self.app.push_screen(SummaryScreen(self.num_chips, self.run_number))
```

- [ ] **Step 7: Remove the old `_launch_chip` method**

Delete the `@work async def _launch_chip(...)` method (lines 813–887). It is entirely replaced by `_launch_model` + `_dispatch_next`. The old method set `TT_VISIBLE_DEVICES=str(chip_id)` with the entire chip queue — the new approach dispatches one model per subprocess.

- [ ] **Step 8: Verify syntax and import**

```bash
python3 -c "import py_compile; py_compile.compile('expedition_tui.py', doraise=True); print('expedition_tui.py syntax OK')"
python3 -c "from expedition_tui import ExpeditionTUI, RunScreen; print('RunScreen import OK')"
```

Expected: Both print `OK`

- [ ] **Step 9: Commit**

```bash
git add expedition_tui.py
git commit -m "feat: RunScreen per-model dispatcher replaces full-queue launch"
```

---

## Task 9: RallyBanner widget + RALLY integration

**Files:**
- Modify: `expedition_tui.py` — add `RallyBanner` widget class, add to `RunScreen.compose()`, add `_fire_rally()` method, add CSS for `#rally-banner`

- [ ] **Step 1: Add `RallyBanner` widget class**

After the `EventLog` class definition (around line 292) and before `SetupScreen`, add:

```python
class RallyBanner(Static):
    """Full-width banner that replaces the chip grid during a RALLY compile.

    Shown when all chips commit to one large model (mesh dispatch). Displays
    model name, chip count, live compile output from the lead chip, and a
    dim status row for each locked chip.
    """

    DEFAULT_CSS = """
    RallyBanner {
        display: none;
        width: 1fr;
        height: 1fr;
        border: double gold;
        padding: 1 2;
        color: $text;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._model_name = ""
        self._chip_ids: list[int] = []
        self._backend = ""
        self._confidence = 0.0

    def start(self, model: dict, chip_ids: list[int], decision) -> None:
        """Activate the banner for the given model and chip configuration."""
        self._model_name = model.get("model_id", "?").split("/")[-1]
        self._chip_ids   = chip_ids
        self._backend    = decision.backend
        self._confidence = decision.confidence
        chips_str = "·".join(str(c) for c in chip_ids)
        self.update(
            f"[bold gold]⚡⚡ RALLY — CHIPS {chips_str} ASSEMBLED ⚡⚡[/]\n"
            f"[dim]{self._model_name}  ·  {len(chip_ids)}-chip mesh  ·  "
            f"{self._backend}  ·  conf {self._confidence:.2f}[/]\n\n"
            f"[green]▶ Compiling on mesh {chips_str}...[/]\n"
        )

    def append_output(self, line: str) -> None:
        """Stream live output from the lead chip into the banner."""
        # Replace the banner content with the last 6 lines of output.
        current = str(self.renderable)
        lines = current.split("\n")
        lines.append(line.rstrip())
        # Keep header (first 4 lines) + last 6 output lines.
        header = lines[:4]
        tail   = lines[4:][-6:]
        self.update("\n".join(header + tail))
```

- [ ] **Step 2: Add `#rally-banner` to `RunScreen.compose()` and CSS**

In `RunScreen.CSS`, add after the existing rules:

```css
    #rally-banner {
        display: none;
        width: 3fr;
        height: 1fr;
    }
```

In `RunScreen.compose()`, after the `with Vertical(id="chip-grid"):` block (around line 784), add `RallyBanner` inside the `#main` Horizontal but outside `#chip-grid`:

```python
        with Horizontal(id="main"):
            with Vertical(id="chip-grid"):
                # ... existing chip rows ...
            yield RallyBanner(id="rally-banner")
            with Vertical(id="sidebar"):
                # ... existing sidebar ...
```

- [ ] **Step 3: Add `_fire_rally()` method to `RunScreen`**

```python
    @work
    async def _fire_rally(self, mesh_model: dict, chip_ids: list[int]) -> None:
        """Handle a RALLY event: show banner, fire mesh subprocess."""
        self._mesh_holding      = None
        self._opportunist_active = False
        for cid in chip_ids:
            self._free_chips.discard(cid)

        decision = mesh_model.get("decision")

        # Show RALLY banner, hide chip grid.
        try:
            self.query_one("#chip-grid").display    = False
            self.query_one("#rally-banner").display = True
            rally = self.query_one("#rally-banner", RallyBanner)
            rally.start(mesh_model, chip_ids, decision)
        except Exception:
            pass

        try:
            el = self.query_one("#event-log", EventLog)
            chips_str = "+".join(str(c) for c in chip_ids)
            el.write(
                f"[bold gold]⚡ RALLY — {mesh_model.get('model_id','?').split('/')[-1]} "
                f"on chips {chips_str}[/]"
            )
        except Exception:
            pass

        # Launch a single multi-chip subprocess on the lead chip.
        # _launch_model handles TT_VISIBLE_DEVICES and formation-share scoring via mesh_chip_ids.
        lead = chip_ids[0]
        self._launch_model(lead, mesh_model, decision, mesh_chip_ids=chip_ids)
```

- [ ] **Step 4: Wire RallyBanner live output into `_launch_model`**

In the `_launch_model` async for loop (where output is streamed to the panel), when `mesh_chip_ids` is provided, also forward lines to the RallyBanner:

```python
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            if panel:
                panel.write_line(line)
            # During a RALLY, also stream output to the rally banner.
            if mesh_chip_ids:
                try:
                    self.query_one("#rally-banner", RallyBanner).append_output(line)
                except Exception:
                    pass
            self._parse_for_events(chip_id, line)
```

- [ ] **Step 5: Verify syntax and import**

```bash
python3 -c "import py_compile; py_compile.compile('expedition_tui.py', doraise=True); print('syntax OK')"
python3 -c "from expedition_tui import ExpeditionTUI, RunScreen; print('import OK')"
```

Expected: Both print `OK`

- [ ] **Step 6: Full syntax check on all changed files**

```bash
python3 -c "
import py_compile
for f in [
    'lib/expedition/router.py',
    'lib/expedition/scorer.py',
    'lib/expedition/bestiary.py',
    'lib/expedition/hf_discover.py',
    'expedition.py',
    'expedition_tui.py',
    'lib/expedition/expedition_worker.py',
    'lib/expedition/expedition_worker_xla.py',
]:
    py_compile.compile(f, doraise=True)
    print('ok', f)
"
```

Expected: 8 lines of `ok <file>`

- [ ] **Step 7: Run full test suite**

```bash
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -v
```

Expected: All tests pass

- [ ] **Step 8: Dry run (no hardware)**

```bash
python3 expedition.py run --seed-only --limit 3 --chips 1 --no-predownload 2>&1 | head -20
```

Expected: No import errors; run begins (may fail at compile step — that's fine, hardware not needed for this check)

- [ ] **Step 9: Commit**

```bash
git add expedition_tui.py
git commit -m "feat: RallyBanner widget + RALLY integration — dramatic 4-chip mesh takeover"
```

---

## Final Verification

```bash
# Router unit tests
python3 -c "
from lib.expedition.router import route_model, DispatchDecision
from lib.expedition.bestiary import Bestiary
b = Bestiary()
d = route_model({'model_id': 'google/flax-bert', 'library': 'jax', 'hf_downloads': 5000, 'mesh_chips': 1}, b)
assert d.backend == 'xla', d
print('router: jax-native → xla  OK')
d = route_model({'model_id': 'deepseek-ai/deepseek-v3', 'library': 'pytorch', 'hf_downloads': 1e6, 'mesh_chips': 4, 'hf_params_b': 67.0}, b, available_chips=set(range(4)))
assert d.chips == 4, d
print('router: large params → 4 chips  OK')
d = route_model({'model_id': 'foo/bar', 'library': 'pytorch', 'hf_downloads': 100, 'mesh_chips': 1}, b)
assert d.backend == 'forge', d
print('router: default → forge  OK')
"

# Scorer mesh_mult + opportunist + formation_share
python3 -c "
from lib.expedition.scorer import compute_score, Rarity, Newness
s1 = compute_score(True, True, Rarity.COMMON, Newness.FRESH, 0, mesh_chips=1)
s4 = compute_score(True, True, Rarity.COMMON, Newness.FRESH, 0, mesh_chips=4)
assert s4.pts == int(s1.pts * 2.5), f'{s4.pts} != {int(s1.pts * 2.5)}'
print(f'scorer: mesh_mult OK  1-chip={s1.pts}  4-chip={s4.pts}')
opp = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, is_opportunist=True)
assert opp.breakdown['opportunist_bonus'] == 25
print('scorer: opportunist bonus OK')
share = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, is_formation_share=True)
assert share.pts == 150
print('scorer: formation share OK')
"

# TUI import
python3 -c "from expedition_tui import ExpeditionTUI; print('TUI import OK')"

# Full test suite
python3 -m pytest tests/expedition/test_intelligent_dispatch.py -v --tb=short
```
