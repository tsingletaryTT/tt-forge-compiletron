# Expedition Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Expedition Mode — a roguelike multi-chip forge compilation runner with chip rivalry scoring, a persistent bestiary, live HuggingFace frontier discovery, per-task inference decode, and a full end-of-run summary screen.

**Architecture:** New `expedition.py` entry point + `lib/expedition/` subpackage alongside existing `compiletron.py` (untouched). Each chip runs `lib/expedition/expedition_worker.py` which wraps the forge compile pipeline and pipes results through decoder → scorer → hud → status file. Persistent state lives in `data/bestiary.json`, `data/artifacts/`, and `data/runs/`.

**Tech Stack:** Python 3.10+, `huggingface_hub`, `pyfiglet`, `torch`, `forge` (tt-forge-fe), existing `lib/hardware.py` / `lib/worker.py` noise-suppression helpers

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `lib/expedition/__init__.py` | Create | Package init, re-exports |
| `lib/expedition/scorer.py` | Create | Rarity/newness enums, points formula |
| `lib/expedition/bestiary.py` | Create | Persistent bestiary.json read/write |
| `lib/expedition/decoder.py` | Create | Per-task inference output decode |
| `lib/expedition/hud.py` | Create | Per-chip score state + status file IPC |
| `lib/expedition/hf_discover.py` | Create | HF Hub query, frontier filtering, dynamic loaders |
| `lib/expedition/expedition_worker.py` | Create | Per-chip worker: compile + decode + score + display |
| `expedition.py` | Create | Entry point: hardware, queue, tmux launch |
| `scripts/run_expedition.sh` | Create | tmux layout for expedition sessions |
| `scripts/status_display.sh` | Modify | Add pts/streak columns for expedition status files |
| `tests/expedition/__init__.py` | Create | Test package |
| `tests/expedition/test_scorer.py` | Create | Scorer unit tests |
| `tests/expedition/test_bestiary.py` | Create | Bestiary persistence tests |
| `tests/expedition/test_decoder.py` | Create | Decoder dispatch tests |
| `tests/expedition/test_hud.py` | Create | HUD state + file write tests |
| `tests/expedition/test_hf_discover.py` | Create | HF discovery + filter tests |

---

## Task 1: Package skeleton + scorer

**Files:**
- Create: `lib/expedition/__init__.py`
- Create: `lib/expedition/scorer.py`
- Create: `tests/expedition/__init__.py`
- Create: `tests/expedition/test_scorer.py`

- [ ] **Step 1: Create test file**

```python
# tests/expedition/test_scorer.py
import pytest
from datetime import datetime, timezone, timedelta
from lib.expedition.scorer import (
    Rarity, Newness, ScoreResult,
    compute_rarity, compute_newness, compute_score,
)


def _dt(days_ago: float) -> str:
    """Return ISO datetime string N days in the past."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


class TestComputeRarity:
    def test_legendary(self):
        assert compute_rarity(15_000_000) == Rarity.LEGENDARY

    def test_rare(self):
        assert compute_rarity(5_000_000) == Rarity.RARE

    def test_uncommon(self):
        assert compute_rarity(500_000) == Rarity.UNCOMMON

    def test_common(self):
        assert compute_rarity(50_000) == Rarity.COMMON

    def test_none_is_familiar(self):
        assert compute_rarity(None) == Rarity.FAMILIAR


class TestComputeNewness:
    def test_zero_day(self):
        assert compute_newness(_dt(0.5), is_first_ever=True) == Newness.ZERO_DAY

    def test_hot(self):
        assert compute_newness(_dt(3), is_first_ever=True) == Newness.HOT

    def test_fresh(self):
        assert compute_newness(_dt(15), is_first_ever=True) == Newness.FRESH

    def test_recent(self):
        assert compute_newness(_dt(60), is_first_ever=True) == Newness.RECENT

    def test_established(self):
        assert compute_newness(_dt(200), is_first_ever=True) == Newness.ESTABLISHED

    def test_not_first_ever_always_established(self):
        assert compute_newness(_dt(0.5), is_first_ever=False) == Newness.ESTABLISHED

    def test_none_date_is_familiar(self):
        assert compute_newness(None, is_first_ever=True) == Newness.FAMILIAR


class TestComputeScore:
    def test_failure(self):
        result = compute_score(success=False, is_first_ever=False,
                               rarity=Rarity.COMMON, newness=Newness.ESTABLISHED,
                               streak=0)
        assert result.pts == -10

    def test_basic_success(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=0)
        assert result.pts == 50

    def test_first_ever_bonus(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.COMMON, newness=Newness.ESTABLISHED,
                               streak=0)
        assert result.pts == 150  # 50 + 100

    def test_rarity_multiplier_legendary(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.LEGENDARY, newness=Newness.ESTABLISHED,
                               streak=0)
        assert result.pts == 600  # (50+100) * 4

    def test_zero_day_multiplier(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.LEGENDARY, newness=Newness.ZERO_DAY,
                               streak=0)
        assert result.pts == 3000  # (50+100) * 4 * 5

    def test_streak_multiplier(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=5)
        # 50 * 1.5 = 75
        assert result.pts == 75

    def test_streak_capped_at_2x(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=100)
        assert result.pts == 100  # 50 * 2.0 capped

    def test_mesh_bonus_4chip(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=0, mesh_chips=4)
        assert result.pts == 100  # 50 + 50 mesh bonus

    def test_mesh_bonus_galaxy(self):
        result = compute_score(success=True, is_first_ever=False,
                               rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
                               streak=0, mesh_chips=32)
        assert result.pts == 250  # 50 + 200 galaxy bonus

    def test_score_result_has_breakdown(self):
        result = compute_score(success=True, is_first_ever=True,
                               rarity=Rarity.RARE, newness=Newness.HOT,
                               streak=3)
        assert "base" in result.breakdown
        assert "first_ever_bonus" in result.breakdown
        assert "rarity_mult" in result.breakdown
        assert "newness_mult" in result.breakdown
        assert "streak_mult" in result.breakdown
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_scorer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError` — scorer doesn't exist yet.

- [ ] **Step 3: Create package init**

```python
# lib/expedition/__init__.py
# Expedition Mode subpackage
```

```python
# tests/expedition/__init__.py
```

- [ ] **Step 4: Implement scorer.py**

```python
# lib/expedition/scorer.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Rarity(str, Enum):
    FAMILIAR = "familiar"
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    LEGENDARY = "legendary"


class Newness(str, Enum):
    ZERO_DAY = "zero_day"
    HOT = "hot"
    FRESH = "fresh"
    RECENT = "recent"
    ESTABLISHED = "established"
    FAMILIAR = "familiar"


@dataclass
class ScoreResult:
    pts: int
    is_first_ever: bool
    rarity: Rarity
    newness: Newness
    streak_at_score: int
    breakdown: dict = field(default_factory=dict)


_RARITY_THRESHOLDS = [
    (10_000_000, Rarity.LEGENDARY),
    (1_000_000,  Rarity.RARE),
    (100_000,    Rarity.UNCOMMON),
    (0,          Rarity.COMMON),
]

_NEWNESS_THRESHOLDS_DAYS = [
    (1,   Newness.ZERO_DAY),
    (7,   Newness.HOT),
    (30,  Newness.FRESH),
    (90,  Newness.RECENT),
]

_RARITY_MULT = {
    Rarity.FAMILIAR:  1.0,
    Rarity.COMMON:    1.0,
    Rarity.UNCOMMON:  1.5,
    Rarity.RARE:      2.0,
    Rarity.LEGENDARY: 4.0,
}

_NEWNESS_MULT = {
    Newness.ZERO_DAY:    5.0,
    Newness.HOT:         3.0,
    Newness.FRESH:       2.0,
    Newness.RECENT:      1.5,
    Newness.ESTABLISHED: 1.0,
    Newness.FAMILIAR:    1.0,
}


def compute_rarity(hf_downloads: int | None) -> Rarity:
    if hf_downloads is None:
        return Rarity.FAMILIAR
    for threshold, rarity in _RARITY_THRESHOLDS:
        if hf_downloads >= threshold:
            return rarity
    return Rarity.COMMON


def compute_newness(hf_created_at: str | None, is_first_ever: bool) -> Newness:
    if not is_first_ever:
        return Newness.ESTABLISHED
    if hf_created_at is None:
        return Newness.FAMILIAR
    try:
        created = datetime.fromisoformat(hf_created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
    except (ValueError, TypeError):
        return Newness.ESTABLISHED
    for max_days, newness in _NEWNESS_THRESHOLDS_DAYS:
        if age_days < max_days:
            return newness
    return Newness.ESTABLISHED


def compute_score(
    success: bool,
    is_first_ever: bool,
    rarity: Rarity,
    newness: Newness,
    streak: int,
    mesh_chips: int = 1,
) -> ScoreResult:
    if not success:
        return ScoreResult(
            pts=-10, is_first_ever=False, rarity=rarity, newness=newness,
            streak_at_score=streak, breakdown={"failure": -10},
        )

    base = 50
    first_ever_bonus = 100 if is_first_ever else 0
    rarity_mult = _RARITY_MULT[rarity]
    newness_mult = _NEWNESS_MULT[newness] if is_first_ever else 1.0
    streak_mult = min(1.0 + streak * 0.1, 2.0)

    mesh_bonus = 0
    if mesh_chips >= 32:
        mesh_bonus = 200
    elif mesh_chips >= 4:
        mesh_bonus = 50
    elif mesh_chips == 2:
        mesh_bonus = 0  # both chips get full points, no extra bonus needed

    pts = int((base + first_ever_bonus) * rarity_mult * newness_mult * streak_mult) + mesh_bonus

    return ScoreResult(
        pts=pts,
        is_first_ever=is_first_ever,
        rarity=rarity,
        newness=newness,
        streak_at_score=streak,
        breakdown={
            "base": base,
            "first_ever_bonus": first_ever_bonus,
            "rarity_mult": rarity_mult,
            "newness_mult": newness_mult,
            "streak_mult": streak_mult,
            "mesh_bonus": mesh_bonus,
        },
    )
```

- [ ] **Step 5: Run tests and confirm pass**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_scorer.py -v
```

Expected: All 17 tests pass.

- [ ] **Step 6: Commit**

```bash
git add lib/expedition/__init__.py lib/expedition/scorer.py \
        tests/expedition/__init__.py tests/expedition/test_scorer.py
git commit -m "feat(expedition): scorer — rarity, newness, points formula"
```

---

## Task 2: Bestiary

**Files:**
- Create: `lib/expedition/bestiary.py`
- Create: `tests/expedition/test_bestiary.py`

- [ ] **Step 1: Write tests**

```python
# tests/expedition/test_bestiary.py
import json
import pytest
from pathlib import Path
from lib.expedition.bestiary import Bestiary, BestiaryEntry, FailedEntry


@pytest.fixture
def tmp_bestiary(tmp_path):
    return Bestiary(path=str(tmp_path / "bestiary.json"))


def _make_entry(model_id="bert/qa", chip=0, run=1) -> BestiaryEntry:
    return BestiaryEntry(
        model_id=model_id,
        first_compiled="2026-04-27T12:00:00",
        first_chip=chip,
        run=run,
        best_time_s=4.2,
        attempts=1,
        successes=1,
        source="forge_models",
        task="question_answering",
        rarity="familiar",
        hf_downloads=None,
        hf_created_at=None,
        artifact="Tenstorrent makes fast chips",
    )


class TestBestiary:
    def test_empty_bestiary_not_compiled(self, tmp_bestiary):
        assert not tmp_bestiary.is_compiled("bert/qa")

    def test_add_compiled(self, tmp_bestiary):
        tmp_bestiary.add_compiled(_make_entry())
        assert tmp_bestiary.is_compiled("bert/qa")

    def test_not_compiled_after_add_different(self, tmp_bestiary):
        tmp_bestiary.add_compiled(_make_entry("bert/qa"))
        assert not tmp_bestiary.is_compiled("resnet50")

    def test_add_failed(self, tmp_bestiary):
        tmp_bestiary.add_failed("mistral-7b", "RuntimeError: shape mismatch", run=1)
        assert tmp_bestiary.is_failed("mistral-7b")

    def test_failed_tracks_attempts(self, tmp_bestiary):
        tmp_bestiary.add_failed("mistral-7b", "err1", run=1)
        tmp_bestiary.add_failed("mistral-7b", "err2", run=2)
        entry = tmp_bestiary._data["failed"]["mistral-7b"]
        assert entry["attempts"] == 2
        assert entry["last_error"] == "err2"

    def test_get_compiled_ids(self, tmp_bestiary):
        tmp_bestiary.add_compiled(_make_entry("bert/qa"))
        tmp_bestiary.add_compiled(_make_entry("resnet50"))
        ids = tmp_bestiary.get_compiled_ids()
        assert "bert/qa" in ids
        assert "resnet50" in ids

    def test_save_and_reload(self, tmp_path):
        path = str(tmp_path / "bestiary.json")
        b1 = Bestiary(path=path)
        b1.add_compiled(_make_entry("bert/qa"))
        b1.save()

        b2 = Bestiary(path=path)
        b2.load()
        assert b2.is_compiled("bert/qa")

    def test_next_run_number_starts_at_1(self, tmp_bestiary):
        assert tmp_bestiary.next_run_number() == 1

    def test_next_run_number_increments(self, tmp_path):
        path = str(tmp_path / "bestiary.json")
        b = Bestiary(path=path)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (runs_dir / "run_001.json").write_text("{}")
        (runs_dir / "run_002.json").write_text("{}")
        b2 = Bestiary(path=path, runs_dir=str(runs_dir))
        assert b2.next_run_number() == 3

    def test_update_chip_totals(self, tmp_bestiary):
        tmp_bestiary.update_chip_totals(chip_id=0, pts=150, is_first_ever=True, streak=3)
        totals = tmp_bestiary._data["chip_totals"]["0"]
        assert totals["pts"] == 150
        assert totals["first_evers"] == 1
        assert totals["best_streak"] == 3

    def test_update_chip_totals_accumulates(self, tmp_bestiary):
        tmp_bestiary.update_chip_totals(chip_id=0, pts=100, is_first_ever=True, streak=2)
        tmp_bestiary.update_chip_totals(chip_id=0, pts=50,  is_first_ever=False, streak=3)
        totals = tmp_bestiary._data["chip_totals"]["0"]
        assert totals["pts"] == 150
        assert totals["first_evers"] == 1
        assert totals["best_streak"] == 3
```

- [ ] **Step 2: Confirm failure**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_bestiary.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Implement bestiary.py**

```python
# lib/expedition/bestiary.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class BestiaryEntry:
    model_id: str
    first_compiled: str
    first_chip: int
    run: int
    best_time_s: float
    attempts: int
    successes: int
    source: str
    task: str
    rarity: str
    hf_downloads: Optional[int]
    hf_created_at: Optional[str]
    artifact: str


@dataclass
class FailedEntry:
    last_error: str
    attempts: int
    run_first_failed: int


_EMPTY = lambda: {"compiled": {}, "failed": {}, "chip_totals": {}}


class Bestiary:
    def __init__(self, path: str = "data/bestiary.json", runs_dir: str = "data/runs"):
        self._path = Path(path)
        self._runs_dir = Path(runs_dir)
        self._data: dict = _EMPTY()
        if self._path.exists():
            self.load()

    def load(self) -> None:
        with open(self._path) as f:
            self._data = json.load(f)
        # Ensure all top-level keys exist (forward-compat with older files)
        for key in ("compiled", "failed", "chip_totals"):
            self._data.setdefault(key, {})

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_compiled(self, model_id: str) -> bool:
        return model_id in self._data["compiled"]

    def is_failed(self, model_id: str) -> bool:
        return model_id in self._data["failed"]

    def get_compiled_ids(self) -> set[str]:
        return set(self._data["compiled"].keys())

    def add_compiled(self, entry: BestiaryEntry) -> None:
        existing = self._data["compiled"].get(entry.model_id)
        if existing:
            existing["attempts"] += 1
            existing["successes"] += 1
            existing["best_time_s"] = min(existing["best_time_s"], entry.best_time_s)
            existing["artifact"] = entry.artifact  # update to latest
        else:
            self._data["compiled"][entry.model_id] = asdict(entry)
        # Remove from failed if it was there
        self._data["failed"].pop(entry.model_id, None)

    def add_failed(self, model_id: str, error: str, run: int) -> None:
        if model_id in self._data["failed"]:
            self._data["failed"][model_id]["attempts"] += 1
            self._data["failed"][model_id]["last_error"] = error
        else:
            self._data["failed"][model_id] = {
                "last_error": error,
                "attempts": 1,
                "run_first_failed": run,
            }

    def update_chip_totals(self, chip_id: int, pts: int, is_first_ever: bool, streak: int) -> None:
        key = str(chip_id)
        if key not in self._data["chip_totals"]:
            self._data["chip_totals"][key] = {"pts": 0, "first_evers": 0, "best_streak": 0}
        totals = self._data["chip_totals"][key]
        totals["pts"] += pts
        if is_first_ever:
            totals["first_evers"] += 1
        totals["best_streak"] = max(totals["best_streak"], streak)

    def next_run_number(self) -> int:
        if not self._runs_dir.exists():
            return 1
        existing = list(self._runs_dir.glob("run_*.json"))
        return len(existing) + 1
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_bestiary.py -v
```

Expected: All 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add lib/expedition/bestiary.py tests/expedition/test_bestiary.py
git commit -m "feat(expedition): bestiary — persistent model history and chip totals"
```

---

## Task 3: Decoder

**Files:**
- Create: `lib/expedition/decoder.py`
- Create: `tests/expedition/test_decoder.py`

- [ ] **Step 1: Write tests**

```python
# tests/expedition/test_decoder.py
import pytest
from unittest.mock import MagicMock
from lib.expedition.decoder import decode, FrontierModelInfo


def _make_tensor(shape, values=None):
    """Return a MagicMock that behaves like a torch tensor for decode purposes."""
    t = MagicMock()
    t.shape = shape
    t.dtype = "torch.float32"
    t.__len__ = lambda self: shape[0]
    if values is not None:
        t.tolist.return_value = values
        t.argmax.return_value = MagicMock(item=lambda: 42)
        t.topk.return_value = (
            MagicMock(tolist=lambda: [0.9, 0.7, 0.5]),
            MagicMock(tolist=lambda: [42, 7, 99]),
        )
    import numpy as np
    t.float.return_value = t
    t.cpu.return_value = t
    t.numpy.return_value = np.zeros(shape)
    return t


class TestDecodeImageClassification:
    def test_returns_string(self):
        info = FrontierModelInfo(name="resnet50", task="image-classification")
        output = _make_tensor((1, 1000))
        output.topk.return_value = (
            MagicMock(tolist=lambda: [0.9, 0.7, 0.5]),
            MagicMock(tolist=lambda: [42, 7, 99]),
        )
        result = decode(output, info)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_shows_confidence(self):
        info = FrontierModelInfo(name="resnet50", task="image-classification")
        output = _make_tensor((1, 1000))
        output.topk.return_value = (
            MagicMock(tolist=lambda: [0.92, 0.71, 0.55]),
            MagicMock(tolist=lambda: [42, 7, 99]),
        )
        result = decode(output, info)
        assert "0.92" in result or "92" in result


class TestDecodeObjectDetection:
    def test_returns_string(self):
        info = FrontierModelInfo(name="yolov8", task="object-detection")
        # Simulate output dict-like with boxes and scores
        output = MagicMock()
        output.shape = (1, 100, 6)
        result = decode(output, info)
        assert isinstance(result, str)


class TestDecodeTextGeneration:
    def test_uses_tokenizer_when_available(self):
        info = FrontierModelInfo(name="gpt2", task="text-generation")
        output = _make_tensor((1, 50, 50257))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Hello world from GPT-2"
        result = decode(output, info, tokenizer=tokenizer)
        assert "Hello world" in result

    def test_falls_back_without_tokenizer(self):
        info = FrontierModelInfo(name="gpt2", task="text-generation")
        output = _make_tensor((1, 50, 50257))
        result = decode(output, info, tokenizer=None)
        assert isinstance(result, str)


class TestDecodeRawFallback:
    def test_unknown_task_returns_shape_info(self):
        info = FrontierModelInfo(name="mystery", task="unknown-task-xyz")
        output = _make_tensor((1, 256, 256))
        result = decode(output, info)
        assert "shape" in result.lower() or "256" in result

    def test_exception_in_decode_returns_fallback(self):
        info = FrontierModelInfo(name="crash", task="image-classification")
        output = MagicMock(spec=[])  # no attributes — will raise on access
        result = decode(output, info)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFrontierModelInfo:
    def test_has_task_attribute(self):
        info = FrontierModelInfo(name="test", task="text-generation")
        assert info.task == "text-generation"
        assert info.source == "huggingface"
```

- [ ] **Step 2: Confirm failure**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_decoder.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Implement decoder.py**

```python
# lib/expedition/decoder.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FrontierModelInfo:
    name: str
    task: str
    source: str = "huggingface"


# Maps pipeline_tag / task string to internal task family
_TASK_FAMILY = {
    "text-generation":             "causal_lm",
    "text2text-generation":        "seq2seq_lm",
    "fill-mask":                   "masked_lm",
    "question-answering":          "qa",
    "image-classification":        "image_cls",
    "object-detection":            "obj_det",
    "semantic-segmentation":       "segmentation",
    "image-segmentation":          "segmentation",
    "depth-estimation":            "depth",
    "automatic-speech-recognition":"asr",
    "audio-classification":        "audio_cls",
    "image-to-text":               "img2text",
    "visual-question-answering":   "img2text",
    "image-captioning":            "img2text",
    "text-to-speech":              "tts",
    "text-to-image":               "img_gen",
    # tt-forge-models ModelTask string forms
    "causal_lm":                   "causal_lm",
    "causal_lm_with_past":         "causal_lm",
    "seq2seq_lm":                  "seq2seq_lm",
    "masked_lm":                   "masked_lm",
    "question_answering":          "qa",
    "image_classification":        "image_cls",
    "object_detection":            "obj_det",
    "semantic_segmentation":       "segmentation",
    "panoptic_segmentation":       "segmentation",
    "depth_estimation":            "depth",
    "automatic_speech_recognition":"asr",
    "audio_classification":        "audio_cls",
    "image_to_text":               "img2text",
    "visual_qa":                   "img2text",
    "image_captioning":            "img2text",
    "text_to_speech":              "tts",
    "image_generation":            "img_gen",
}


def _raw_fallback(output) -> str:
    try:
        if hasattr(output, "shape"):
            shape = tuple(output.shape)
            try:
                import numpy as np
                arr = output.float().cpu().numpy()
                mn, mx = float(arr.min()), float(arr.max())
                return f"shape={shape} dtype={getattr(output, 'dtype', '?')} range=[{mn:.2f}, {mx:.2f}]"
            except Exception:
                return f"shape={shape}"
        return f"output={type(output).__name__}"
    except Exception:
        return "decode failed — raw output"


def _decode_causal_lm(output, tokenizer) -> str:
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        if hasattr(output, "shape") and len(output.shape) == 3:
            import torch
            token_ids = output[0].argmax(dim=-1).tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return text[:100] if text else _raw_fallback(output)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_image_cls(output) -> str:
    try:
        if hasattr(output, "topk"):
            scores, indices = output.topk(3)
            score_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            idx_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            if isinstance(score_list[0], list):
                score_list = score_list[0]
                idx_list = idx_list[0]
            parts = [f"class_{idx} {score:.2f}" for idx, score in zip(idx_list, score_list)]
            return "top-3: " + ", ".join(parts)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_obj_det(output) -> str:
    try:
        if hasattr(output, "shape"):
            shape = tuple(output.shape)
            return f"detection output shape={shape}"
        if isinstance(output, (list, tuple)) and len(output) > 0:
            return f"{len(output)} detection(s) in output"
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_segmentation(output) -> str:
    try:
        if hasattr(output, "shape"):
            return f"segmentation map shape={tuple(output.shape)}"
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_depth(output) -> str:
    try:
        if hasattr(output, "shape"):
            try:
                import numpy as np
                arr = output.float().cpu().numpy()
                return f"depth map shape={tuple(output.shape)} range=[{arr.min():.2f}m, {arr.max():.2f}m]"
            except Exception:
                return f"depth map shape={tuple(output.shape)}"
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_asr(output, tokenizer) -> str:
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        if hasattr(output, "shape") and len(output.shape) == 2:
            import torch
            ids = output[0].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return text[:100] if text else _raw_fallback(output)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


_FAMILY_DECODERS = {
    "causal_lm":   lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "seq2seq_lm":  lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "masked_lm":   lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "qa":          lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "image_cls":   lambda out, _tok, _inp: _decode_image_cls(out),
    "obj_det":     lambda out, _tok, _inp: _decode_obj_det(out),
    "segmentation":lambda out, _tok, _inp: _decode_segmentation(out),
    "depth":       lambda out, _tok, _inp: _decode_depth(out),
    "asr":         lambda out, tok, _inp: _decode_asr(out, tok),
    "audio_cls":   lambda out, _tok, _inp: _decode_image_cls(out),
    "img2text":    lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "tts":         lambda out, _tok, _inp: f"audio output shape={tuple(out.shape) if hasattr(out, 'shape') else '?'}",
    "img_gen":     lambda out, _tok, _inp: f"image output shape={tuple(out.shape) if hasattr(out, 'shape') else '?'}",
}


def decode(
    output: Any,
    model_info,
    inputs: Any = None,
    tokenizer: Any = None,
) -> str:
    try:
        task_str = getattr(model_info, "task", "") or ""
        family = _TASK_FAMILY.get(task_str.lower(), None)
        if family and family in _FAMILY_DECODERS:
            return _FAMILY_DECODERS[family](output, tokenizer, inputs)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_decoder.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add lib/expedition/decoder.py tests/expedition/test_decoder.py
git commit -m "feat(expedition): decoder — per-task inference output decode with raw fallback"
```

---

## Task 4: HUD

**Files:**
- Create: `lib/expedition/hud.py`
- Create: `tests/expedition/test_hud.py`

- [ ] **Step 1: Write tests**

```python
# tests/expedition/test_hud.py
import os
import pytest
from lib.expedition.scorer import ScoreResult, Rarity, Newness
from lib.expedition.hud import ChipHUD, ChipState


def _score(pts: int, first_ever: bool = False) -> ScoreResult:
    return ScoreResult(
        pts=pts, is_first_ever=first_ever,
        rarity=Rarity.FAMILIAR, newness=Newness.ESTABLISHED,
        streak_at_score=0,
    )


@pytest.fixture
def hud(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPEDITION_STATUS_DIR", str(tmp_path))
    return ChipHUD(chip_id=0, total_models=10)


class TestChipHUD:
    def test_initial_state(self, hud):
        assert hud.state.pts == 0
        assert hud.state.streak == 0
        assert hud.state.successes == 0
        assert hud.state.failures == 0

    def test_record_success_increments_pts(self, hud):
        hud.record_success("bert/qa", _score(150, first_ever=True))
        assert hud.state.pts == 150
        assert hud.state.successes == 1

    def test_record_success_increments_streak(self, hud):
        hud.record_success("bert/qa", _score(50))
        hud.record_success("resnet50", _score(50))
        assert hud.state.streak == 2

    def test_record_failure_resets_streak(self, hud):
        hud.record_success("bert/qa", _score(50))
        hud.record_success("resnet50", _score(50))
        hud.record_failure("mistral-7b")
        assert hud.state.streak == 0
        assert hud.state.failures == 1

    def test_best_streak_preserved_after_reset(self, hud):
        hud.record_success("a", _score(50))
        hud.record_success("b", _score(50))
        hud.record_success("c", _score(50))
        hud.record_failure("d")
        assert hud.state.best_streak == 3

    def test_failure_deducts_pts(self, hud):
        hud.record_success("bert/qa", _score(100))
        hud.record_failure("mistral")
        assert hud.state.pts == 90

    def test_set_current(self, hud):
        hud.set_current("bert/qa", index=3)
        assert hud.state.current_model == "bert/qa"
        assert hud.state.current_index == 3

    def test_write_status_creates_file(self, hud, tmp_path):
        hud.set_current("bert/qa", index=1)
        hud.record_success("bert/qa", _score(50))
        hud.write_status()
        status_file = tmp_path / "expedition_chip_0.status"
        assert status_file.exists()
        content = status_file.read_text()
        assert "pts=50" in content
        assert "chip_id=0" in content

    def test_write_status_includes_streak(self, hud, tmp_path):
        hud.record_success("a", _score(50))
        hud.record_success("b", _score(50))
        hud.write_status()
        content = (tmp_path / "expedition_chip_0.status").read_text()
        assert "streak=2" in content

    def test_write_status_includes_done_flag(self, hud, tmp_path):
        hud.mark_done()
        hud.write_status()
        content = (tmp_path / "expedition_chip_0.status").read_text()
        assert "done=1" in content
```

- [ ] **Step 2: Confirm failure**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_hud.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement hud.py**

```python
# lib/expedition/hud.py
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from lib.expedition.scorer import ScoreResult


_STATUS_DIR_ENV = "EXPEDITION_STATUS_DIR"
_DEFAULT_STATUS_DIR = "/tmp"


@dataclass
class ChipState:
    chip_id: int
    pts: int = 0
    streak: int = 0
    best_streak: int = 0
    successes: int = 0
    failures: int = 0
    current_model: str = ""
    current_index: int = 0
    total_models: int = 0
    done: bool = False


class ChipHUD:
    def __init__(self, chip_id: int, total_models: int):
        self._state = ChipState(chip_id=chip_id, total_models=total_models)

    @property
    def state(self) -> ChipState:
        return self._state

    def set_current(self, model_id: str, index: int) -> None:
        self._state.current_model = model_id
        self._state.current_index = index

    def record_success(self, model_id: str, score: ScoreResult) -> None:
        self._state.pts += score.pts
        self._state.successes += 1
        self._state.streak += 1
        self._state.best_streak = max(self._state.best_streak, self._state.streak)

    def record_failure(self, model_id: str) -> None:
        self._state.pts -= 10
        self._state.failures += 1
        self._state.streak = 0

    def mark_done(self) -> None:
        self._state.done = True

    def write_status(self) -> None:
        status_dir = os.environ.get(_STATUS_DIR_ENV, _DEFAULT_STATUS_DIR)
        path = Path(status_dir) / f"expedition_chip_{self._state.chip_id}.status"
        s = self._state
        lines = [
            f"chip_id={s.chip_id}",
            f"current={s.current_index}",
            f"total={s.total_models}",
            f"successes={s.successes}",
            f"failures={s.failures}",
            f"pts={s.pts}",
            f"streak={s.streak}",
            f"best_streak={s.best_streak}",
            f"model={s.current_model}",
            f"done={1 if s.done else 0}",
        ]
        path.write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_hud.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add lib/expedition/hud.py tests/expedition/test_hud.py
git commit -m "feat(expedition): hud — per-chip score state and status file IPC"
```

---

## Task 5: HF Discovery

**Files:**
- Create: `lib/expedition/hf_discover.py`
- Create: `tests/expedition/test_hf_discover.py`

- [ ] **Step 1: Write tests**

```python
# tests/expedition/test_hf_discover.py
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from lib.expedition.hf_discover import (
    discover_frontier, build_dynamic_loader, FrontierModel,
    _model_to_frontier,
)
from lib.expedition.scorer import Rarity, Newness


def _mock_model(model_id="org/model", pipeline_tag="text-generation",
                downloads=500_000, days_ago=60):
    m = MagicMock()
    m.id = model_id
    m.pipeline_tag = pipeline_tag
    m.downloads = downloads
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    m.created_at = created
    return m


class TestModelToFrontier:
    def test_basic_conversion(self):
        mock = _mock_model("org/bert", "text-classification", downloads=200_000, days_ago=45)
        result = _model_to_frontier(mock)
        assert result.model_id == "org/bert"
        assert result.pipeline_tag == "text-classification"
        assert result.rarity == Rarity.UNCOMMON
        assert result.newness == Newness.RECENT

    def test_legendary_rarity(self):
        mock = _mock_model(downloads=20_000_000)
        result = _model_to_frontier(mock)
        assert result.rarity == Rarity.LEGENDARY

    def test_zero_day_newness(self):
        mock = _mock_model(days_ago=0.25)
        result = _model_to_frontier(mock)
        assert result.newness == Newness.ZERO_DAY


class TestDiscoverFrontier:
    def test_filters_already_compiled(self):
        models = [
            _mock_model("org/bert"),
            _mock_model("org/gpt2"),
        ]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(
                compiled_ids={"org/bert"},
                known_model_ids=set(),
            )
        ids = [m.model_id for m in result]
        assert "org/bert" not in ids
        assert "org/gpt2" in ids

    def test_filters_known_forge_models(self):
        models = [_mock_model("org/resnet50")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(
                compiled_ids=set(),
                known_model_ids={"org/resnet50"},
            )
        assert len(result) == 0

    def test_skips_unsupported_pipeline_tag(self):
        models = [_mock_model("org/weird", pipeline_tag="reinforcement-learning")]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0

    def test_skips_none_pipeline_tag(self):
        models = [_mock_model("org/notag", pipeline_tag=None)]
        with patch("lib.expedition.hf_discover.HfApi") as MockApi:
            MockApi.return_value.list_models.return_value = iter(models)
            result = discover_frontier(compiled_ids=set(), known_model_ids=set())
        assert len(result) == 0


class TestBuildDynamicLoader:
    def test_returns_callable_for_text_generation(self):
        model = FrontierModel(
            model_id="gpt2",
            pipeline_tag="text-generation",
            downloads=1_000_000,
            created_at=None,
            rarity=Rarity.RARE,
            newness=Newness.ESTABLISHED,
        )
        loader = build_dynamic_loader(model)
        assert loader is not None
        assert callable(loader)

    def test_returns_none_for_unsupported_tag(self):
        model = FrontierModel(
            model_id="org/rl",
            pipeline_tag="reinforcement-learning",
            downloads=100,
            created_at=None,
            rarity=Rarity.COMMON,
            newness=Newness.ESTABLISHED,
        )
        result = build_dynamic_loader(model)
        assert result is None
```

- [ ] **Step 2: Confirm failure**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_hf_discover.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement hf_discover.py**

```python
# lib/expedition/hf_discover.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable
from lib.expedition.scorer import (
    Rarity, Newness, compute_rarity, compute_newness,
)

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


_SUPPORTED_TAGS = {
    "text-generation", "text2text-generation", "fill-mask",
    "question-answering", "image-classification", "object-detection",
    "semantic-segmentation", "image-segmentation", "depth-estimation",
    "automatic-speech-recognition", "audio-classification",
    "image-to-text", "visual-question-answering", "image-captioning",
    "text-to-speech", "text-to-image",
}

# pipeline_tag → (AutoModel class name, dummy input description)
_TAG_TO_AUTO = {
    "text-generation":             ("AutoModelForCausalLM",         "text"),
    "text2text-generation":        ("AutoModelForSeq2SeqLM",        "text"),
    "fill-mask":                   ("AutoModelForMaskedLM",         "text"),
    "question-answering":          ("AutoModelForQuestionAnswering","text"),
    "image-classification":        ("AutoModelForImageClassification","image"),
    "object-detection":            ("AutoModelForObjectDetection",   "image"),
    "semantic-segmentation":       ("AutoModelForSemanticSegmentation","image"),
    "image-segmentation":          ("AutoModelForImageSegmentation", "image"),
    "depth-estimation":            ("AutoModelForDepthEstimation",   "image"),
    "automatic-speech-recognition":("AutoModelForSpeechSeq2Seq",    "audio"),
    "audio-classification":        ("AutoModelForAudioClassification","audio"),
    "image-to-text":               ("AutoModelForVision2Seq",        "image"),
    "visual-question-answering":   ("AutoModelForVision2Seq",        "image"),
    "image-captioning":            ("AutoModelForVision2Seq",        "image"),
}

_LARGE_MOE_PATTERNS = ["deepseek", "mixtral", "qwen", "kimi"]


@dataclass
class FrontierModel:
    model_id: str
    pipeline_tag: str
    downloads: int
    created_at: Optional[datetime]
    rarity: Rarity
    newness: Newness
    mesh_chips: int = 1


def _model_to_frontier(hf_model) -> FrontierModel:
    created_at = getattr(hf_model, "created_at", None)
    created_str = created_at.isoformat() if created_at else None
    downloads = getattr(hf_model, "downloads", 0) or 0
    rarity = compute_rarity(downloads)
    newness = compute_newness(created_str, is_first_ever=True)

    # Heuristic: large MoE models likely need multi-chip
    model_id_lower = hf_model.id.lower()
    mesh_chips = 4 if any(p in model_id_lower for p in _LARGE_MOE_PATTERNS) else 1

    return FrontierModel(
        model_id=hf_model.id,
        pipeline_tag=hf_model.pipeline_tag or "",
        downloads=downloads,
        created_at=created_at,
        rarity=rarity,
        newness=newness,
        mesh_chips=mesh_chips,
    )


def discover_frontier(
    compiled_ids: set[str],
    known_model_ids: set[str],
    limit: int = 500,
) -> list[FrontierModel]:
    if HfApi is None:
        return []

    api = HfApi()
    try:
        hf_models = api.list_models(
            filter="pytorch",
            sort="createdAt",
            direction=-1,
            limit=limit,
        )
    except Exception:
        return []

    results = []
    for m in hf_models:
        tag = getattr(m, "pipeline_tag", None)
        if not tag or tag not in _SUPPORTED_TAGS:
            continue
        if m.id in compiled_ids or m.id in known_model_ids:
            continue
        results.append(_model_to_frontier(m))

    return results


def build_dynamic_loader(model: FrontierModel) -> Optional[Callable]:
    tag = model.pipeline_tag
    if tag not in _TAG_TO_AUTO:
        return None
    auto_class_name, input_type = _TAG_TO_AUTO[tag]
    model_id = model.model_id

    def loader():
        import transformers
        AutoClass = getattr(transformers, auto_class_name, None)
        if AutoClass is None:
            raise ImportError(f"transformers.{auto_class_name} not found")
        return AutoClass.from_pretrained(model_id)

    loader.__name__ = f"load_{model_id.replace('/', '_')}"
    loader._input_type = input_type
    loader._model_id = model_id
    return loader
```

- [ ] **Step 4: Run tests**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/test_hf_discover.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add lib/expedition/hf_discover.py tests/expedition/test_hf_discover.py
git commit -m "feat(expedition): hf_discover — frontier model query, filtering, dynamic loaders"
```

---

## Task 6: Expedition Worker

**Files:**
- Create: `lib/expedition/expedition_worker.py`

No dedicated unit tests — this is integration-level (requires forge). Functional tests happen in Task 9 via a live run. Verify correct import + argument parsing only.

- [ ] **Step 1: Implement expedition_worker.py**

```python
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
PURPLE = "\033[95m"
PINK   = "\033[95m"
BLUE   = "\033[94m"
GOLD   = "\033[33m"
DIM    = "\033[2m"


# ── Rarity display config ────────────────────────────────────────────────────

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


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def _decouple_stderr():
    """Silence fd2 for C++ noise — same as lib/worker.py."""
    import sys, os
    # Add project root so `lib.worker` is importable when this script is run
    # as `python3 lib/expedition/expedition_worker.py` (cwd != project root)
    _project_root = str(Path(__file__).parent.parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from lib.worker import FilteredStderr
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
    import pyfiglet

    rarity_color, rarity_label, rarity_pause = _RARITY_STYLE.get(
        rarity, (CYAN, "", 0.0))
    newness_color, newness_label, newness_pause = _NEWNESS_STYLE.get(
        newness, ("", "", 0.0))

    pause = max(rarity_pause, newness_pause)

    # Badges
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

    # Model name banner
    short_name = model_id.split("/")[-1]
    font = "small" if len(short_name) > 25 else "standard"
    try:
        banner = pyfiglet.figlet_format(short_name, font=font)
        print(f"{rarity_color or CYAN}{banner}{RESET}", end="")
    except Exception:
        print(f"\n{BOLD}{rarity_color or CYAN}  {short_name}{RESET}\n")

    meta_parts = [task, source]
    print(f"  {DIM}{' · '.join(p for p in meta_parts if p)}{RESET}")

    if pause > 0:
        time.sleep(pause)


def _print_progress_step(step: int, total: int, desc: str, color=YELLOW) -> None:
    print(f"  {color}[{step}/{total}]{RESET} {desc}")


def _print_live_info(msg: str, ok: bool = True) -> None:
    marker = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}→{RESET}"
    print(f"    {marker} {msg}")


def _print_success(model_id: str, compile_time: float, total_time: float,
                   artifact: str, score_pts: int, is_first_ever: bool,
                   streak: int) -> None:
    print(f"\n  {BOLD}{GREEN}✓ SUCCESS{RESET}")
    print(f"    compile: {compile_time:.1f}s  total: {total_time:.1f}s  "
          f"pts: {GOLD}{score_pts:+d}{RESET}"
          + (f"  {GOLD}★ FIRST EVER{RESET}" if is_first_ever else "")
          + (f"  🔥×{streak}" if streak >= 2 else ""))
    if artifact:
        print(f"    {CYAN}❝ {artifact[:120]}{RESET}")


def _print_failure(model_id: str, error: str, elapsed: float) -> None:
    print(f"\n  {BOLD}{RED}✗ FAILED{RESET}  {DIM}{error[:80]}{RESET}  ({elapsed:.1f}s  −10pts)")


def _compile_model(model_loader, chip_id: int, timeout: int = 120) -> tuple[bool, Any, float, str]:
    """
    Run forge compile + inference. Returns (success, output, compile_time, error_str).
    Imports are deferred so this module can be imported without forge installed.
    """
    import torch

    sys.path.insert(0, os.path.expanduser("~/tt-forge-fe"))
    import forge

    try:
        model = model_loader()
        model.eval()

        # Determine input shape heuristically from the loader if available,
        # otherwise use a generic image shape
        if hasattr(model_loader, "_input_type"):
            itype = model_loader._input_type
        else:
            itype = "image"

        if itype == "text":
            sample_input = torch.randint(0, 1000, (1, 32))
        elif itype == "audio":
            sample_input = torch.randn(1, 16000)
        else:
            sample_input = torch.randn(1, 3, 224, 224)

        _print_live_info(f"Architecture: {type(model).__name__}")

        compile_start = time.time()
        _print_progress_step(2, 3, "Compiling for TT hardware...")
        compiled = forge.compile(model, sample_inputs=[sample_input])
        compile_time = time.time() - compile_start

        _print_progress_step(3, 3, f"Running inference on chip {chip_id}...")
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            output = compiled(sample_input)
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            return False, None, compile_time, "TIMEOUT"

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
    model_id: str
    display_name: str
    task: str
    source: str
    rarity: str
    hf_downloads: Optional[int]
    hf_created_at: Optional[str]
    mesh_chips: int
    loader_module: Optional[str]  # e.g. "bert.question_answering.pytorch.loader"
    loader_class: Optional[str]   # e.g. "BertQALoader"
    is_frontier: bool = False


def _load_queue(queue_path: str) -> list[QueueItem]:
    with open(queue_path) as f:
        items = json.load(f)
    return [QueueItem(**item) for item in items]


def _build_loader(item: QueueItem):
    """Return a callable that loads the model."""
    if item.is_frontier:
        # Dynamic HF loader
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
        # tt-forge-models loader
        import importlib
        forge_models_path = os.path.expanduser("~/code/tt-forge-models")
        if forge_models_path not in sys.path:
            sys.path.insert(0, forge_models_path)
        mod = importlib.import_module(item.loader_module)
        cls = getattr(mod, item.loader_class)
        instance = cls()
        def loader():
            return instance.load_model()
        loader._input_type = "image"  # will be overridden by instance if available
        return loader


def run_worker(chip_id: int, run_number: int, bestiary_path: str,
               queue_path: str, results_path: str) -> None:
    from lib.expedition.bestiary import Bestiary, BestiaryEntry
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

    results = []
    last_artifact = ""

    for idx, item in enumerate(queue, 1):
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

        if last_artifact:
            print(f"  {DIM}last: {last_artifact[:80]}{RESET}")

        _print_progress_step(1, 3, "Loading model...")
        start = time.time()

        try:
            loader = _build_loader(item)
        except Exception as e:
            elapsed = time.time() - start
            _print_failure(item.model_id, str(e), elapsed)
            score = compute_score(False, is_first_ever, rarity, newness, hud.state.streak,
                                  mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            bestiary.add_failed(item.model_id, str(e), run_number)
            hud.write_status()
            results.append({"model": item.model_id, "status": "failed",
                            "error": str(e), "pts": score.pts})
            continue

        success, output, compile_time, error_str = _compile_model(loader, chip_id)
        elapsed = time.time() - start

        if success:
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

            # Save artifact to data/artifacts/
            artifact_key = item.model_id.replace("/", "__")
            artifact_file = Path("data/artifacts") / f"{artifact_key}.txt"
            artifact_file.parent.mkdir(parents=True, exist_ok=True)
            artifact_file.write_text(
                f"model={item.model_id} task={item.task} source={item.source} "
                f"chip={chip_id} run={run_number}\n{artifact}\n"
            )

            entry = BestiaryEntry(
                model_id=item.model_id,
                first_compiled=__import__("datetime").datetime.now().isoformat(),
                first_chip=chip_id,
                run=run_number,
                best_time_s=compile_time,
                attempts=1,
                successes=1,
                source=item.source,
                task=item.task,
                rarity=rarity.value,
                hf_downloads=item.hf_downloads,
                hf_created_at=item.hf_created_at,
                artifact=artifact,
            )
            bestiary.add_compiled(entry)
            bestiary.update_chip_totals(chip_id, score.pts, is_first_ever, hud.state.streak)
            results.append({"model": item.model_id, "status": "success",
                            "pts": score.pts, "compile_time": compile_time,
                            "artifact": artifact, "first_ever": is_first_ever})
        else:
            _print_failure(item.model_id, error_str, elapsed)
            score = compute_score(False, is_first_ever, rarity, newness,
                                  hud.state.streak, mesh_chips=item.mesh_chips)
            hud.record_failure(item.model_id)
            bestiary.add_failed(item.model_id, error_str, run_number)
            results.append({"model": item.model_id, "status": "failed",
                            "error": error_str, "pts": score.pts})

        bestiary.save()
        hud.write_status()

    hud.mark_done()
    hud.write_status()

    # Write results CSV
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    s = hud.state
    print(f"\n{BOLD}{GREEN}{'═'*80}{RESET}")
    print(f"{BOLD}CHIP {chip_id} DONE{RESET}  pts:{GOLD}{s.pts}{RESET}  "
          f"✓{s.successes} ✗{s.failures}  best streak: 🔥×{s.best_streak}")
    print(f"{BOLD}{GREEN}{'═'*80}{RESET}")
    input("Press Enter to close...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", type=int, required=True)
    parser.add_argument("--run",  type=int, required=True)
    parser.add_argument("--bestiary", default="data/bestiary.json")
    parser.add_argument("--queue",    required=True)
    parser.add_argument("--results",  required=True)
    args = parser.parse_args()

    run_worker(
        chip_id=args.chip,
        run_number=args.run,
        bestiary_path=args.bestiary,
        queue_path=args.queue,
        results_path=args.results,
    )
```

- [ ] **Step 2: Verify importability**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 -c "from lib.expedition.expedition_worker import run_worker; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "feat(expedition): expedition_worker — per-chip compile/decode/score pipeline"
```

---

## Task 7: Entry Point + Queue Builder

**Files:**
- Create: `expedition.py`

- [ ] **Step 1: Implement expedition.py**

```python
#!/usr/bin/env python3
# expedition.py
"""
Expedition Mode entry point.

Usage:
  python3 expedition.py                        # auto-detect chips, full run
  python3 expedition.py --chips 2              # limit to 2 chips
  python3 expedition.py --seed-only            # skip HF discovery
  python3 expedition.py --frontier-only        # skip forge-models seed
  python3 expedition.py --limit 20             # cap models per chip
  python3 expedition.py summary                # print bestiary summary
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
BESTIARY_PATH = DATA_DIR / "bestiary.json"
RUNS_DIR = DATA_DIR / "runs"
ARTIFACTS_DIR = DATA_DIR / "artifacts"


# ── Queue building ────────────────────────────────────────────────────────────

def _scan_forge_models(bestiary_compiled_ids: set[str]) -> list[dict]:
    """
    Walk ~/code/tt-forge-models and return QueueItem dicts for loaders
    not yet in the bestiary.
    """
    forge_models_root = Path.home() / "code" / "tt-forge-models"
    if not forge_models_root.exists():
        return []

    sys.path.insert(0, str(forge_models_root))
    items = []

    for loader_py in sorted(forge_models_root.rglob("loader.py")):
        # Skip __pycache__ and hidden dirs
        if any(p.startswith("_") or p.startswith(".") for p in loader_py.parts):
            continue

        # Derive a stable model_id from the path relative to forge_models_root
        rel = loader_py.relative_to(forge_models_root)
        model_id = "/".join(rel.parts[:-1])  # e.g. "bert/question_answering/pytorch"

        if model_id in bestiary_compiled_ids:
            continue

        # Derive module path for importlib
        module_path = ".".join(rel.parts[:-1]) + ".loader"

        # Try to find the loader class name
        try:
            import importlib
            mod = importlib.import_module(module_path)
            # Find the ForgeModel subclass
            from base import ForgeModel
            cls_name = None
            for name in dir(mod):
                obj = getattr(mod, name)
                try:
                    if isinstance(obj, type) and issubclass(obj, ForgeModel) and obj is not ForgeModel:
                        cls_name = name
                        break
                except Exception:
                    continue
            if cls_name is None:
                continue

            # Get model_info to extract task/source
            instance = obj()
            info = instance._get_model_info()
            task = info.task.value if hasattr(info.task, "value") else str(info.task)
            source = info.source.value if hasattr(info.source, "value") else str(info.source)

        except Exception:
            # Loader import failed — skip gracefully
            continue

        items.append({
            "model_id": model_id,
            "display_name": model_id.split("/")[0].replace("_", " ").title(),
            "task": task,
            "source": source,
            "rarity": "familiar",
            "hf_downloads": None,
            "hf_created_at": None,
            "mesh_chips": 1,
            "loader_module": module_path,
            "loader_class": cls_name,
            "is_frontier": False,
        })

    return items


def _scan_frontier(bestiary_compiled_ids: set[str], forge_model_ids: set[str]) -> list[dict]:
    from lib.expedition.hf_discover import discover_frontier, FrontierModel
    models = discover_frontier(
        compiled_ids=bestiary_compiled_ids,
        known_model_ids=forge_model_ids,
    )
    return [
        {
            "model_id": m.model_id,
            "display_name": m.model_id.split("/")[-1],
            "task": m.pipeline_tag,
            "source": "huggingface",
            "rarity": m.rarity.value,
            "hf_downloads": m.downloads,
            "hf_created_at": m.created_at.isoformat() if m.created_at else None,
            "mesh_chips": m.mesh_chips,
            "loader_module": None,
            "loader_class": None,
            "is_frontier": True,
        }
        for m in models
    ]


def build_queues(
    num_chips: int,
    seed_only: bool = False,
    frontier_only: bool = False,
    limit_per_chip: int = 0,
) -> list[list[dict]]:
    from lib.expedition.bestiary import Bestiary
    bestiary = Bestiary(path=str(BESTIARY_PATH), runs_dir=str(RUNS_DIR))
    compiled_ids = bestiary.get_compiled_ids()

    seed_items: list[dict] = []
    frontier_items: list[dict] = []

    if not frontier_only:
        print("  Scanning tt-forge-models library...")
        seed_items = _scan_forge_models(compiled_ids)
        print(f"  {len(seed_items)} seed models queued (not yet compiled)")

    forge_ids = {item["model_id"] for item in seed_items}

    if not seed_only:
        print("  Querying HuggingFace frontier...")
        frontier_items = _scan_frontier(compiled_ids, forge_ids)
        print(f"  {len(frontier_items)} frontier models discovered")

    # Interleave: 60% seed, 40% frontier in round-robin order
    all_items = _interleave(seed_items, frontier_items, seed_ratio=0.6)
    print(f"  Total queue: {len(all_items)} models across {num_chips} chip(s)")

    # Distribute round-robin across chips
    chip_queues: list[list[dict]] = [[] for _ in range(num_chips)]
    for i, item in enumerate(all_items):
        chip_queues[i % num_chips].append(item)

    if limit_per_chip > 0:
        chip_queues = [q[:limit_per_chip] for q in chip_queues]

    return chip_queues


def _interleave(seed: list, frontier: list, seed_ratio: float) -> list:
    result = []
    si = fi = 0
    seed_budget = 0.0
    while si < len(seed) or fi < len(frontier):
        seed_budget += seed_ratio
        while seed_budget >= 1.0 and si < len(seed):
            result.append(seed[si]); si += 1; seed_budget -= 1.0
        if fi < len(frontier):
            result.append(frontier[fi]); fi += 1
    return result


# ── Run summary ──────────────────────────────────────────────────────────────

def _print_run_summary(num_chips: int, run_number: int) -> None:
    """Aggregate end-of-run summary printed to the launching terminal after tmux exits."""
    import csv
    from lib.expedition.bestiary import Bestiary

    chip_results: list[dict] = []
    for chip_id in range(num_chips):
        path = Path(f"/tmp/expedition_results_chip{chip_id}.csv")
        if not path.exists():
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        successes = [r for r in rows if r.get("status") == "success"]
        failures  = [r for r in rows if r.get("status") == "failed"]
        total_pts = sum(int(r.get("pts", 0)) for r in rows)
        first_evers = [r for r in successes if r.get("first_ever") == "True"]
        chip_results.append({
            "chip_id": chip_id,
            "pts": total_pts,
            "successes": successes,
            "failures": failures,
            "first_evers": first_evers,
        })

    chip_results.sort(key=lambda x: -x["pts"])

    medals = ["🥇", "🥈", "🥉", "  "]
    W = 72
    print(f"\n{'═'*W}")
    print(f"  EXPEDITION #{run_number:03d} COMPLETE")
    print(f"{'═'*W}")
    for i, c in enumerate(chip_results):
        medal = medals[min(i, 3)]
        fe = len(c["first_evers"])
        print(f"  {medal} CHIP {c['chip_id']}   {c['pts']:,} pts   "
              f"✓{len(c['successes'])} ✗{len(c['failures'])}   ★{fe} first-evers")

    # New bestiary entries this run
    all_first_evers = [
        r for c in chip_results for r in c["first_evers"]
    ]
    if all_first_evers:
        print(f"\n{'─'*W}")
        print("  NEW TO BESTIARY:")
        for r in all_first_evers:
            artifact = (r.get("artifact") or "")[:80]
            rune = "★"
            print(f"  {rune} {r['model']:40s}  {artifact}")

    all_failures = [r for c in chip_results for r in c["failures"]]
    if all_failures:
        print(f"\n{'─'*W}")
        print("  FAILED:")
        for r in all_failures:
            print(f"  ✗ {r['model']:40s}  {(r.get('error') or '')[:40]}")

    b = Bestiary(path=str(BESTIARY_PATH), runs_dir=str(RUNS_DIR))
    compiled_count = len(b.get_compiled_ids())
    print(f"\n{'─'*W}")
    print(f"  BESTIARY: {compiled_count} total compiled")
    print(f"{'═'*W}\n")

    # Write run record
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_file = RUNS_DIR / f"run_{run_number:03d}.json"
    run_file.write_text(json.dumps({
        "run": run_number,
        "timestamp": datetime.now().isoformat(),
        "chips": num_chips,
        "chip_results": [
            {k: v if not isinstance(v, list) else len(v) for k, v in c.items()}
            for c in chip_results
        ],
        "new_bestiary_entries": len(all_first_evers),
        "total_failures": len(all_failures),
    }, indent=2))


# ── Summary command ───────────────────────────────────────────────────────────

def cmd_summary():
    from lib.expedition.bestiary import Bestiary
    b = Bestiary(path=str(BESTIARY_PATH), runs_dir=str(RUNS_DIR))
    compiled = b._data.get("compiled", {})
    failed = b._data.get("failed", {})
    totals = b._data.get("chip_totals", {})

    print(f"\n{'═'*60}")
    print(f"  EXPEDITION BESTIARY")
    print(f"{'═'*60}")
    print(f"  Compiled:  {len(compiled)} models")
    print(f"  Failed:    {len(failed)} models")
    if totals:
        print(f"\n  Chip Hall of Fame:")
        for chip_id, data in sorted(totals.items(), key=lambda x: -x[1]["pts"]):
            print(f"    Chip {chip_id}: {data['pts']:,} pts  "
                  f"★{data['first_evers']} first-evers  "
                  f"best streak ×{data['best_streak']}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Expedition Mode — roguelike forge compilation")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Start an expedition run (default)")
    run_p.add_argument("--chips",          type=int, default=0,
                       help="Number of chips (0=auto-detect)")
    run_p.add_argument("--limit",          type=int, default=0,
                       help="Max models per chip (0=unlimited)")
    run_p.add_argument("--seed-only",      action="store_true")
    run_p.add_argument("--frontier-only",  action="store_true")

    sub.add_parser("summary", help="Print bestiary summary")

    args = parser.parse_args()

    if args.cmd == "summary" or (args.cmd is None and len(sys.argv) == 1):
        if args.cmd == "summary":
            cmd_summary(); return
        # Default: run
        args.cmd = "run"
        args.chips = 0
        args.limit = 0
        args.seed_only = False
        args.frontier_only = False

    # Hardware detection
    from lib.hardware import detect_hardware, get_hardware_summary
    hw = detect_hardware()
    num_chips = args.chips if args.chips > 0 else hw.get("num_chips", 1)
    if num_chips == 0:
        print("No chips detected. Check tt-smi.")
        sys.exit(1)
    print(f"\n  Hardware: {get_hardware_summary(hw)}")
    print(f"  Chips for this run: {num_chips}")

    # Bestiary + run number
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    from lib.expedition.bestiary import Bestiary
    bestiary = Bestiary(path=str(BESTIARY_PATH), runs_dir=str(RUNS_DIR))
    run_number = bestiary.next_run_number()
    print(f"  Run #{run_number:03d}")

    # Build queues
    chip_queues = build_queues(
        num_chips=num_chips,
        seed_only=args.seed_only,
        frontier_only=args.frontier_only,
        limit_per_chip=args.limit,
    )

    # Write per-chip queue JSON to /tmp
    for chip_id, queue in enumerate(chip_queues):
        queue_path = f"/tmp/expedition_queue_chip{chip_id}.json"
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)
        print(f"  Chip {chip_id}: {len(queue)} models → {queue_path}")

    # Launch tmux — blocks until user detaches or session ends
    script = PROJECT_DIR / "scripts" / "run_expedition.sh"
    env = {**os.environ, "EXPEDITION_RUN": str(run_number),
           "EXPEDITION_NUM_CHIPS": str(num_chips)}
    subprocess.run(["bash", str(script), "--chips", str(num_chips),
                    "--run", str(run_number)], env=env)

    # After tmux exits, gather results from per-chip CSVs and print aggregate summary
    _print_run_summary(num_chips, run_number)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify importability**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 -c "import expedition; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Verify summary command (no hardware needed)**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 expedition.py summary
```

Expected: Prints bestiary summary (empty on first run — that's fine).

- [ ] **Step 4: Commit**

```bash
git add expedition.py
git commit -m "feat(expedition): expedition.py — entry point, queue builder, summary command"
```

---

## Task 8: Tmux Layout Script

**Files:**
- Create: `scripts/run_expedition.sh`

- [ ] **Step 1: Implement run_expedition.sh**

```bash
#!/bin/bash
# scripts/run_expedition.sh
# Expedition Mode tmux layout — 4 chip panes + shared status strip
#
# Layout (identical to run_4way_tmux.sh):
#   ┌──────────────┬──────────────┐
#   │  Chip 0      │  Chip 1      │
#   ├──────────────┼──────────────┤
#   │  Chip 2      │  Chip 3      │
#   ├──────────────┴──────────────┤
#   │  Status (scores, streaks)   │
#   └─────────────────────────────┘
#
# Invoked by expedition.py with:
#   bash scripts/run_expedition.sh --chips N --run R

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="expedition"
NUM_CHIPS=4
RUN_NUMBER=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chips) NUM_CHIPS="$2"; shift 2 ;;
        --run)   RUN_NUMBER="$2"; shift 2 ;;
        *)       shift ;;
    esac
done

if ! command -v tmux &>/dev/null; then
    echo "✗ tmux not installed: sudo apt install tmux"; exit 1
fi

if [[ ! -f ~/tt-forge-fe/env/activate ]]; then
    echo "✗ ~/tt-forge-fe/env/activate not found"; exit 1
fi

# Clear stale expedition status files
rm -f /tmp/expedition_chip_{0,1,2,3}.status

# Write per-chip launcher scripts to /tmp
for chip_id in $(seq 0 $((NUM_CHIPS - 1))); do
    stagger=$((chip_id * 4))
    cat > "/tmp/expedition_chip_${chip_id}.sh" << CHIPSCRIPT
#!/bin/bash
clear
echo "┌─────────────────────────────────────"
echo "│  EXPEDITION  Chip ${chip_id}  Run #$(printf '%03d' ${RUN_NUMBER})"
echo "│"
echo ""

source ~/tt-forge-fe/env/activate
if [[ -z "\${TTFORGE_TOOLCHAIN_DIR}" && -z "\${TTMLIR_TOOLCHAIN_DIR}" ]]; then
    echo "ERROR: Forge env activation failed."
    read -rp "Press Enter to close..."
    exit 1
fi

STAGGER=${stagger}
if [[ \$STAGGER -gt 0 ]]; then
    echo "⏳ Staggered start: waiting \${STAGGER}s..."
    sleep \$STAGGER
fi

export TT_VISIBLE_DEVICES=${chip_id}
export TT_METAL_ARCH_NAME=blackhole
export TT_MESH_GRAPH_DESC_PATH=${PROJECT_DIR}/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto

python3 ${PROJECT_DIR}/lib/expedition/expedition_worker.py \
    --chip ${chip_id} \
    --run ${RUN_NUMBER} \
    --bestiary ${PROJECT_DIR}/data/bestiary.json \
    --queue /tmp/expedition_queue_chip${chip_id}.json \
    --results /tmp/expedition_results_chip${chip_id}.csv
CHIPSCRIPT
    chmod +x "/tmp/expedition_chip_${chip_id}.sh"
done

# Build tmux layout
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"

P_TL=$(tmux display-message -t "$SESSION" -p "#{pane_id}")
P_STA=$(tmux split-window -v -l 6 -t "$P_TL" -P -F "#{pane_id}")

if [[ "$NUM_CHIPS" -ge 2 ]]; then
    P_TR=$(tmux split-window -h -l 50% -t "$P_TL" -P -F "#{pane_id}")
fi
if [[ "$NUM_CHIPS" -ge 3 ]]; then
    P_BL=$(tmux split-window -v -l 50% -t "$P_TL" -P -F "#{pane_id}")
fi
if [[ "$NUM_CHIPS" -ge 4 ]]; then
    P_BR=$(tmux split-window -v -l 50% -t "$P_TR" -P -F "#{pane_id}")
fi

# Pane titles
tmux select-pane -t "$P_TL"  -T "  Chip 0 — Expedition #$(printf '%03d' $RUN_NUMBER)  "
[[ -n "$P_TR"  ]] && tmux select-pane -t "$P_TR"  -T "  Chip 1  "
[[ -n "$P_BL"  ]] && tmux select-pane -t "$P_BL"  -T "  Chip 2  "
[[ -n "$P_BR"  ]] && tmux select-pane -t "$P_BR"  -T "  Chip 3  "
tmux select-pane -t "$P_STA" -T "  Score Board  "

tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "
tmux set -t "$SESSION" pane-border-style "fg=colour240"
tmux set -t "$SESSION" pane-active-border-style "fg=colour214,bold"

# Launch workers
tmux send-keys -t "$P_TL" "bash /tmp/expedition_chip_0.sh" C-m
[[ -n "$P_TR"  ]] && tmux send-keys -t "$P_TR"  "bash /tmp/expedition_chip_1.sh" C-m
[[ -n "$P_BL"  ]] && tmux send-keys -t "$P_BL"  "bash /tmp/expedition_chip_2.sh" C-m
[[ -n "$P_BR"  ]] && tmux send-keys -t "$P_BR"  "bash /tmp/expedition_chip_3.sh" C-m

# Status strip — reads expedition_chip_N.status files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmux send-keys -t "$P_STA" \
    "watch -n1 -t '${SCRIPT_DIR}/status_display.sh --expedition'" \
    C-m

tmux select-pane -t "$P_TL"
echo ""
echo "  Expedition Run #$(printf '%03d' $RUN_NUMBER) — $NUM_CHIPS chip(s)"
echo ""
echo "  Ctrl+B + arrow = navigate  |  Ctrl+B + D = detach"
echo ""
tmux attach-session -t "$SESSION"
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x /home/ttuser/code/tt-forge-compiletron/scripts/run_expedition.sh
bash -n /home/ttuser/code/tt-forge-compiletron/scripts/run_expedition.sh
```

Expected: No output (clean syntax check).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_expedition.sh
git commit -m "feat(expedition): run_expedition.sh — tmux layout for expedition sessions"
```

---

## Task 9: Extend Status Display + End-of-Run Summary

**Files:**
- Modify: `scripts/status_display.sh`

- [ ] **Step 1: Read current status_display.sh**

Read `scripts/status_display.sh` (already read above — it's 68 lines, reads `compiletron_chip_N.status` files).

- [ ] **Step 2: Add expedition mode to status_display.sh**

```bash
# Add after the existing shebang and color definitions, before render_bar:
```

Edit `scripts/status_display.sh` — add expedition mode support by detecting which status files exist and adding `--expedition` flag handling. The full updated file:

```bash
#!/usr/bin/env bash
# Renders per-chip ASCII progress bars for the bottom status strip.
# Called by `watch -n1` from run_4way_tmux.sh or run_expedition.sh.
#
# Classic mode: reads /tmp/compiletron_chip_N.status
# Expedition mode (--expedition): reads /tmp/expedition_chip_N.status
#
# Classic format:  chip_id= current= total= successes= failures= model= done=
# Expedition adds: pts= streak= best_streak=

BOLD=$'\033[1m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
RED=$'\033[31m'
CYAN=$'\033[36m'
YELLOW=$'\033[33m'
PURPLE=$'\033[35m'
GOLD=$'\033[33m'
DIM=$'\033[2m'

BAR_LEN=24
MODE="classic"
[[ "$1" == "--expedition" ]] && MODE="expedition"

render_bar_classic() {
    local current=$1 total=$2 successes=$3 failures=$4 model=$5 done=$6

    if [[ "$total" -le 0 ]]; then
        printf "%s[%s] waiting...%s\n" "$DIM" "$(printf '░%.0s' $(seq 1 $BAR_LEN))" "$RESET"
        return
    fi

    local filled=$(( BAR_LEN * current / total ))
    local empty=$(( BAR_LEN - filled ))
    local pct=$(( 100 * current / total ))
    local bar
    bar="$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))$(printf '░%.0s' $(seq 1 $empty 2>/dev/null))"
    local stats="${GREEN}✓${successes}${RESET}/${RED}✗${failures}${RESET}"
    local label="${model:0:22}"

    if [[ "$done" == "1" ]]; then
        printf "%s[%s]%s %3d%% %s  %s✓ DONE%s\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$stats" "$GREEN" "$RESET"
    else
        printf "%s[%s]%s %3d%% %s  %s%s%s\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$stats" "$CYAN" "$label" "$RESET"
    fi
}

render_bar_expedition() {
    local current=$1 total=$2 successes=$3 failures=$4 model=$5 done=$6 pts=$7 streak=$8

    if [[ "$total" -le 0 ]]; then
        printf "%s[%s] waiting...%s\n" "$DIM" "$(printf '░%.0s' $(seq 1 $BAR_LEN))" "$RESET"
        return
    fi

    local filled=$(( BAR_LEN * current / total ))
    local empty=$(( BAR_LEN - filled ))
    local pct=$(( 100 * current / total ))
    local bar
    bar="$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))$(printf '░%.0s' $(seq 1 $empty 2>/dev/null))"

    local label="${model:0:18}"
    local streak_str=""
    if [[ "$streak" -ge 2 ]]; then
        streak_str=" 🔥×${streak}"
    fi

    if [[ "$done" == "1" ]]; then
        printf "%s[%s]%s %3d%% ${GREEN}✓${successes}${RESET}/${RED}✗${failures}${RESET}  pts:${GOLD}%s${RESET}  ${GREEN}✓ DONE${RESET}\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$pts"
    else
        printf "%s[%s]%s %3d%% ${GREEN}✓${successes}${RESET}/${RED}✗${failures}${RESET}  pts:${GOLD}%s${RESET}%s  ${CYAN}%s${RESET}\n" \
            "$BOLD" "$bar" "$RESET" "$pct" "$pts" "$streak_str" "$label"
    fi
}

# Print one line per chip
for chip in 0 1 2 3; do
    if [[ "$MODE" == "expedition" ]]; then
        file="/tmp/expedition_chip_${chip}.status"
    else
        file="/tmp/compiletron_chip_${chip}.status"
    fi

    if [[ -f "$file" ]]; then
        chip_id=$(grep '^chip_id='    "$file" | cut -d= -f2)
        current=$(grep '^current='    "$file" | cut -d= -f2)
        total=$(grep '^total='        "$file" | cut -d= -f2)
        succ=$(grep '^successes='     "$file" | cut -d= -f2)
        fail=$(grep '^failures='      "$file" | cut -d= -f2)
        model=$(grep '^model='        "$file" | cut -d= -f2-)
        done=$(grep '^done='          "$file" | cut -d= -f2)
        pts=$(grep '^pts='            "$file" | cut -d= -f2)
        streak=$(grep '^streak='      "$file" | cut -d= -f2)

        printf "${BOLD}${YELLOW}C%d${RESET} " "$chip"
        if [[ "$MODE" == "expedition" ]]; then
            render_bar_expedition "$current" "$total" "$succ" "$fail" "$model" "$done" "$pts" "$streak"
        else
            render_bar_classic "$current" "$total" "$succ" "$fail" "$model" "$done"
        fi
    else
        printf "${BOLD}${YELLOW}C%d${RESET} ${DIM}[%-${BAR_LEN}s] waiting for worker...${RESET}\n" \
            "$chip" ""
    fi
done
```

- [ ] **Step 3: Verify syntax**

```bash
bash -n /home/ttuser/code/tt-forge-compiletron/scripts/status_display.sh
```

Expected: No output.

- [ ] **Step 4: Commit**

```bash
git add scripts/status_display.sh
git commit -m "feat(expedition): status_display.sh — add pts and streak columns for expedition mode"
```

---

## Task 10: Wire Everything + Smoke Test

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/expedition/ -v
```

Expected: All tests pass (scorer, bestiary, decoder, hud, hf_discover).

- [ ] **Step 2: Verify expedition.py summary**

```bash
python3 expedition.py summary
```

Expected: Prints empty bestiary summary (0 models compiled, 0 failed).

- [ ] **Step 3: Verify expedition.py --help**

```bash
python3 expedition.py run --help
```

Expected: Help text showing --chips, --limit, --seed-only, --frontier-only flags.

- [ ] **Step 4: Verify queue build (no hardware)**

```bash
python3 expedition.py run --chips 1 --seed-only --limit 3 2>&1 | head -20
```

Expected: Scans forge-models, prints queue size, writes `/tmp/expedition_queue_chip0.json`, then fails on tmux (acceptable — no hardware run yet).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat(expedition): expedition mode complete — bestiary, scoring, HF discovery, worker, tmux"
```

---

## Quick Reference: Running an Expedition

```bash
# Full run (auto-detects chips, full model set)
python3 expedition.py

# Quick test run — 1 chip, 5 models, seed only (no HF)
python3 expedition.py run --chips 1 --limit 5 --seed-only

# Frontier only — hunt for new HF models
python3 expedition.py run --frontier-only

# Check your bestiary
python3 expedition.py summary
```
