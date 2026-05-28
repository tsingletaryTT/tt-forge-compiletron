# Isolated Overlay Envs + Dependency Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every model a disposable thin-venv overlay so dep installs never corrupt the base forge/XLA env, track which packages are missing across the fleet, and surface that data in a ranked report.

**Architecture:** A new `lib/expedition/model_overlay.py` module owns overlay create/install/destroy. The forge and XLA dispatch paths each create an overlay before launching a subprocess and destroy it in a `finally` block. The bestiary gains `pip_deps` (on success) and `missing_packages` (on failure) fields plus a `missing_dep_report()` aggregation method. Two standalone scripts handle fleet reporting and one-shot stale-entry cleanup.

**Tech Stack:** Python `venv` stdlib (overlay creation), `subprocess` (pip installs), `shutil` (cleanup), `pathlib`, existing `Bestiary` class, `multiprocessing.Process` (existing forge subprocess).

---

## File Map

| File | Role |
|---|---|
| `lib/expedition/model_overlay.py` | **New.** Overlay create/install/destroy. Single responsibility. |
| `lib/expedition/expedition_worker.py` | **Modify.** Wire overlay into dispatch loop + XLA dispatch. Add `_pull_tt_forge_models()` and `_find_seed_requirements()`. Remove `_CUSTOM_DEP_MAP` as a reject gate. |
| `lib/expedition/bestiary.py` | **Modify.** Add `pip_deps` to `record_success()`, `missing_packages` to `record_failure()`, new `missing_dep_report()` method. |
| `scripts/missing_deps_report.py` | **New.** CLI tool: ranked table of packages blocking models. |
| `scripts/clean_bestiary.py` | **New.** One-shot cleanup of stale harness-caused bestiary entries. |
| `docs/overlay-deps.md` | **New.** Operator guide. |
| `CLAUDE.md` | **Modify.** Add harness section for overlay + new scripts. |
| `tests/test_model_overlay.py` | **New.** Overlay lifecycle + requirements parsing tests. |
| `tests/test_bestiary_overlay.py` | **New.** pip_deps / missing_packages / report tests. |

---

## Task 1: `model_overlay.py` — overlay create and destroy

**Files:**
- Create: `lib/expedition/model_overlay.py`
- Create: `tests/test_model_overlay.py`

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/test_model_overlay.py
import pathlib, pytest
from lib.expedition.model_overlay import create_overlay, destroy_overlay, ModelOverlay


def test_create_overlay_produces_valid_venv(tmp_path):
    """create_overlay returns a ModelOverlay whose python binary exists."""
    base = pathlib.Path("/opt/ttforge-toolchain/venv")
    if not base.exists():
        pytest.skip("forge base venv not present")
    overlay = create_overlay("test/model", base_venv=base, overlay_root=tmp_path)
    assert isinstance(overlay, ModelOverlay)
    assert overlay.python.exists()
    assert overlay.path.is_dir()
    destroy_overlay(overlay)
    assert not overlay.path.exists()


def test_create_overlay_uses_tmp_by_default():
    """Without overlay_root, overlay lands under /tmp."""
    base = pathlib.Path("/opt/ttforge-toolchain/venv")
    if not base.exists():
        pytest.skip("forge base venv not present")
    overlay = create_overlay("gliner/pytorch", base_venv=base)
    assert str(overlay.path).startswith("/tmp/")
    destroy_overlay(overlay)


def test_destroy_overlay_is_idempotent(tmp_path):
    """destroy_overlay on a non-existent path does not raise."""
    overlay = ModelOverlay(
        path=tmp_path / "nonexistent",
        python=tmp_path / "nonexistent" / "bin" / "python3",
        base_venv=pathlib.Path("/opt/ttforge-toolchain/venv"),
    )
    destroy_overlay(overlay)  # must not raise
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_model_overlay.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'create_overlay'`

- [ ] **Step 1.3: Implement `model_overlay.py`**

```python
# lib/expedition/model_overlay.py
"""Thin per-model venv overlay — isolates pip installs from the base forge/XLA env.

Each model gets a symlinked venv (~10ms to create) that inherits the base env's
packages via --system-site-packages. Installs go into the overlay only. The overlay
is destroyed after the model finishes (success or fail). The base env is never touched.
"""
from __future__ import annotations
import shutil, venv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelOverlay:
    path: Path       # e.g. /tmp/compiletron-overlay-gliner_pytorch_abc123
    python: Path     # path/bin/python3
    base_venv: Path  # the venv this overlays


def create_overlay(
    model_id: str,
    base_venv: Path,
    overlay_root: Path | None = None,
) -> ModelOverlay:
    """Create a thin symlinked venv overlay for one model run.

    Args:
        model_id:     HuggingFace model identifier (slashes replaced with underscores
                      for the directory name).
        base_venv:    Path to the base forge or XLA venv to overlay.
        overlay_root: Parent directory for the overlay. Defaults to /tmp.

    Returns:
        ModelOverlay with path and python interpreter ready to use.
    """
    import tempfile
    safe_name = model_id.replace("/", "_")
    if overlay_root is None:
        overlay_root = Path(tempfile.gettempdir())
    overlay_path = overlay_root / f"compiletron-overlay-{safe_name}"
    # Remove any stale overlay from a previous crashed run.
    if overlay_path.exists():
        shutil.rmtree(overlay_path, ignore_errors=True)
    venv.create(
        str(overlay_path),
        system_site_packages=True,
        symlinks=True,
        with_pip=False,
    )
    python = overlay_path / "bin" / "python3"
    return ModelOverlay(path=overlay_path, python=python, base_venv=base_venv)


def destroy_overlay(overlay: ModelOverlay) -> None:
    """Delete the overlay directory. Silent if already gone."""
    shutil.rmtree(overlay.path, ignore_errors=True)
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
python3 -m pytest tests/test_model_overlay.py -v 2>&1 | tail -10
```

Expected: all 3 PASS

- [ ] **Step 1.5: Commit**

```bash
git add lib/expedition/model_overlay.py tests/test_model_overlay.py
git commit -m "feat: ModelOverlay — thin per-model venv overlay create/destroy"
```

---

## Task 2: `model_overlay.py` — requirements.txt parsing and install

**Files:**
- Modify: `lib/expedition/model_overlay.py`
- Modify: `tests/test_model_overlay.py`

- [ ] **Step 2.1: Write failing tests for requirements parsing**

Append to `tests/test_model_overlay.py`:

```python
from lib.expedition.model_overlay import _parse_requirements, install_requirements


def test_parse_requirements_keeps_plain_packages(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("gliner\nFlagEmbedding\nomegaconf>=2.3.0\n")
    result = _parse_requirements(req)
    assert result == ["gliner", "FlagEmbedding", "omegaconf>=2.3.0"]


def test_parse_requirements_skips_unsafe_lines(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "gliner\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "git+https://github.com/some/repo.git\n"
        "-r other_requirements.txt\n"
        "# just a comment\n"
        "\n"
        "torch==2.5.1\n"           # protected: torch
        "transformers>=4.0\n"       # protected: transformers
        "kornia\n"
    )
    result = _parse_requirements(req)
    assert result == ["gliner", "kornia"]


def test_parse_requirements_empty_file(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("# only comments\n\n")
    assert _parse_requirements(req) == []


def test_install_requirements_calls_pip(tmp_path, monkeypatch):
    """install_requirements invokes pip for each parsed package."""
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    base = pathlib.Path("/opt/ttforge-toolchain/venv")
    overlay = ModelOverlay(
        path=tmp_path / "overlay",
        python=tmp_path / "overlay" / "bin" / "python3",
        base_venv=base,
    )
    req = tmp_path / "requirements.txt"
    req.write_text("gliner\nFlagEmbedding\n")

    installed = install_requirements(overlay, req)
    assert installed == ["gliner", "FlagEmbedding"]
    assert len(calls) == 1  # single pip install -r call
    assert str(overlay.python) in calls[0]
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
python3 -m pytest tests/test_model_overlay.py::test_parse_requirements_keeps_plain_packages -xvs 2>&1 | tail -5
```

Expected: `ImportError: cannot import name '_parse_requirements'`

- [ ] **Step 2.3: Implement `_parse_requirements` and `install_requirements`**

Append to `lib/expedition/model_overlay.py` (after `destroy_overlay`):

```python
# Core packages that must never be downgraded — skip any requirements.txt line
# that names one of these as the top-level package.
_PROTECTED_PACKAGES = frozenset({
    "torch", "torchvision", "torchaudio",
    "transformers", "forge", "tt_lib", "ttnn",
    "jax", "jaxlib", "flax",
})


def _parse_requirements(req_file: Path) -> list[str]:
    """Parse a requirements.txt and return safe-to-install package specs.

    Skips:
      - Comment lines and blank lines
      - Lines starting with -- (index URLs, flags)
      - Lines starting with -r (recursive includes)
      - Lines containing :// (git+https, VCS deps)
      - Lines whose top-level package name is in _PROTECTED_PACKAGES

    Returns list of package specs suitable for passing to pip install.
    """
    pkgs: list[str] = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--") or line.startswith("-r"):
            continue
        if "://" in line:
            continue
        # Extract top-level package name (before any version specifier or extras)
        top = line.split("[")[0].split("=")[0].split(">")[0].split("<")[0].split("!")[0].strip()
        top_lower = top.lower().replace("-", "_")
        if top_lower in {p.lower().replace("-", "_") for p in _PROTECTED_PACKAGES}:
            continue
        pkgs.append(line)
    return pkgs


def install_requirements(overlay: ModelOverlay, req_file: Path) -> list[str]:
    """Parse req_file and pip-install safe packages into the overlay.

    Runs a single `pip install <pkg1> <pkg2> ...` call for all safe packages.
    Fails open — install errors are printed but do not raise.

    Returns list of package specs that were passed to pip (regardless of
    whether install succeeded).
    """
    import subprocess
    pkgs = _parse_requirements(req_file)
    if not pkgs:
        return []
    try:
        subprocess.run(
            [str(overlay.python), "-m", "pip", "install", "-q", "--no-build-isolation",
             *pkgs],
            timeout=120,
            check=False,
        )
    except Exception as exc:
        print(f"  ⚠ overlay pip install failed: {exc}")
    return pkgs
```

- [ ] **Step 2.4: Run all overlay tests**

```bash
python3 -m pytest tests/test_model_overlay.py -v 2>&1 | tail -15
```

Expected: all 7 PASS

- [ ] **Step 2.5: Commit**

```bash
git add lib/expedition/model_overlay.py tests/test_model_overlay.py
git commit -m "feat: overlay requirements parsing + install (protected-package filter)"
```

---

## Task 3: Bestiary — `pip_deps`, `missing_packages`, `missing_dep_report()`

**Files:**
- Modify: `lib/expedition/bestiary.py`
- Create: `tests/test_bestiary_overlay.py`

`record_success()` is at line ~310; `record_failure()` is at line ~426. `missing_dep_report()` will be a new method after `clear_stale_env_failures()`.

- [ ] **Step 3.1: Write failing tests**

```python
# tests/test_bestiary_overlay.py
import json, pathlib, pytest
from lib.expedition.bestiary import Bestiary


def _make_bestiary(tmp_path, compiled=None, failed=None) -> Bestiary:
    data = {
        "compiled": compiled or {},
        "failed": failed or {},
        "chip_totals": {},
    }
    p = tmp_path / "bestiary.json"
    p.write_text(json.dumps(data))
    return Bestiary(path=p)


def test_record_success_stores_pip_deps(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success(
        model_id="gliner/pytorch", chip=0, run=1, time_s=10.0,
        task="token-classification", source="seed", rarity="uncommon",
        hf_downloads=0, hf_created_at="", artifact="", backend="forge",
        pip_deps=["gliner"],
    )
    assert b.compiled["gliner/pytorch"]["pip_deps"] == ["gliner"]


def test_record_success_omits_pip_deps_when_empty(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_success(
        model_id="alexnet/pytorch", chip=0, run=1, time_s=5.0,
        task="image-classification", source="seed", rarity="common",
        hf_downloads=0, hf_created_at="", artifact="", backend="forge",
    )
    assert "pip_deps" not in b.compiled["alexnet/pytorch"]


def test_record_failure_stores_missing_packages(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_failure("surya/pytorch", run=1, error="No module named 'surya'",
                     missing_packages=["surya-ocr"])
    assert b.failed["surya/pytorch"]["missing_packages"] == ["surya-ocr"]


def test_record_failure_merges_missing_packages(tmp_path):
    """Second failure accumulates packages rather than overwriting."""
    b = _make_bestiary(tmp_path, failed={
        "model/x": {
            "run_first_failed": 1,
            "attempts": 1,
            "last_error": "No module named 'alpha'",
            "error_category": "missing_dependency",
            "missing_packages": ["alpha"],
        }
    })
    b.record_failure("model/x", run=2, error="No module named 'beta'",
                     missing_packages=["beta"])
    assert set(b.failed["model/x"]["missing_packages"]) == {"alpha", "beta"}


def test_record_failure_omits_missing_packages_when_empty(tmp_path):
    b = _make_bestiary(tmp_path)
    b.record_failure("model/y", run=1, error="SIGSEGV")
    assert "missing_packages" not in b.failed["model/y"]


def test_missing_dep_report_ranks_by_count(tmp_path):
    b = _make_bestiary(tmp_path, failed={
        "model/a": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["surya-ocr"]},
        "model/b": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["surya-ocr", "torchaudio"]},
        "model/c": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["torchaudio"]},
        "model/d": {"last_error": "", "error_category": "missing_dependency",
                    "attempts": 1, "missing_packages": ["gliner"]},
    })
    report = b.missing_dep_report()
    assert report[0]["package"] == "surya-ocr"
    assert report[0]["count"] == 2
    assert report[1]["package"] == "torchaudio"
    assert report[1]["count"] == 2
    assert report[2]["package"] == "gliner"
    assert report[2]["count"] == 1
    # Each entry has package, count, models keys
    assert "models" in report[0]
    assert "model/a" in report[0]["models"]


def test_missing_dep_report_empty_bestiary(tmp_path):
    b = _make_bestiary(tmp_path)
    assert b.missing_dep_report() == []
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
python3 -m pytest tests/test_bestiary_overlay.py -v 2>&1 | tail -10
```

Expected: `TypeError` (unexpected keyword argument `pip_deps`) or similar.

- [ ] **Step 3.3: Add `pip_deps` to `record_success()`**

In `lib/expedition/bestiary.py`, the `record_success()` signature is around line 310. Add `pip_deps: list[str] = ()` kwarg and store it:

```python
def record_success(self, model_id: str, chip: str, run: int, time_s: float,
                   task: str, source: str, rarity: str, hf_downloads: int,
                   hf_created_at: str, artifact: str, backend: str,
                   first_voice: str = "", compile_s: float = 0.0,
                   infer_s: float = 0.0, throughput: float = 0.0,
                   throughput_unit: str = "", mesh_chips: int = 1,
                   pip_deps: list[str] = ()) -> None:
```

Inside `record_success()`, after the initial entry dict is built (around line 381 where `self._data["compiled"][model_id] = {...}`), add before the closing brace:

```python
        if pip_deps:
            self._data["compiled"][model_id]["pip_deps"] = list(pip_deps)
```

And after `entry = self._data["compiled"][model_id]` (around line 382), update existing entries too:

```python
    if pip_deps and "pip_deps" not in entry:
        entry["pip_deps"] = list(pip_deps)
```

- [ ] **Step 3.4: Add `missing_packages` to `record_failure()`**

`record_failure()` signature is around line 426. Add `missing_packages: list[str] = ()`:

```python
def record_failure(self, model_id: str, run: int, error: str,
                   env_fingerprint: dict[str, str] | None = None,
                   missing_packages: list[str] = ()) -> None:
```

Inside `record_failure()`, after `entry["error_category"] = _classify_error(error)[0]` (around line 453), add:

```python
        if missing_packages:
            existing = entry.get("missing_packages", [])
            entry["missing_packages"] = list(set(existing) | set(missing_packages))
```

- [ ] **Step 3.5: Add `missing_dep_report()` method**

After `clear_stale_env_failures()` in `lib/expedition/bestiary.py`, add:

```python
    def missing_dep_report(self) -> list[dict]:
        """Return a ranked list of packages blocking failed models.

        Aggregates the missing_packages field across all failed entries.
        Returns list of dicts sorted descending by count:
            [{"package": str, "count": int, "models": list[str]}, ...]
        """
        from collections import defaultdict
        tally: dict[str, list[str]] = defaultdict(list)
        for mid, entry in self._data["failed"].items():
            for pkg in entry.get("missing_packages", []):
                tally[pkg].append(mid)
        return sorted(
            [{"package": pkg, "count": len(models), "models": models}
             for pkg, models in tally.items()],
            key=lambda x: x["count"],
            reverse=True,
        )
```

- [ ] **Step 3.6: Run all bestiary overlay tests**

```bash
python3 -m pytest tests/test_bestiary_overlay.py -v 2>&1 | tail -15
```

Expected: all 7 PASS

- [ ] **Step 3.7: Run full test suite to check for regressions**

```bash
python3 -m pytest tests/test_bestiary_envfix.py tests/test_bestiary_overlay.py tests/test_model_overlay.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 3.8: Commit**

```bash
git add lib/expedition/bestiary.py tests/test_bestiary_overlay.py
git commit -m "feat: bestiary pip_deps + missing_packages fields + missing_dep_report()"
```

---

## Task 4: `_pull_tt_forge_models()` + `_find_seed_requirements()`

**Files:**
- Modify: `lib/expedition/expedition_worker.py`
- Modify: `tests/test_model_overlay.py`

- [ ] **Step 4.1: Write failing tests**

Append to `tests/test_model_overlay.py`:

```python
from lib.expedition.expedition_worker import _find_seed_requirements


def test_find_seed_requirements_returns_path_when_exists():
    """Returns path for a seed model that has requirements.txt in tt-forge-models."""
    import pathlib
    # gliner/pytorch has requirements.txt containing "gliner"
    result = _find_seed_requirements("gliner/pytorch")
    if result is None:
        pytest.skip("tt-forge-models not present at ~/code/tt-forge-models")
    assert result.exists()
    assert result.name == "requirements.txt"
    assert "gliner" in result.read_text()


def test_find_seed_requirements_returns_none_for_unknown():
    """Returns None when no requirements.txt exists for the model."""
    result = _find_seed_requirements("nonexistent/model")
    assert result is None


def test_find_seed_requirements_returns_none_for_frontier():
    """Frontier models (no prefix match in tt-forge-models) return None."""
    # A HuggingFace model ID with no matching tt-forge-models directory.
    result = _find_seed_requirements("facebook/opt-125m")
    assert result is None
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
python3 -m pytest tests/test_model_overlay.py::test_find_seed_requirements_returns_none_for_unknown -xvs 2>&1 | tail -5
```

Expected: `ImportError: cannot import name '_find_seed_requirements'`

- [ ] **Step 4.3: Add `_pull_tt_forge_models()` and `_find_seed_requirements()` to `expedition_worker.py`**

Add both functions before `_warm_hf_datasets()` (around line 659):

```python
_TT_FORGE_MODELS_PATH = pathlib.Path.home() / "code" / "tt-forge-models"


def _pull_tt_forge_models() -> None:
    """git pull --ff-only on ~/code/tt-forge-models to keep requirements.txt current.

    Fails open — a stale tt-forge-models tree is better than a blocked expedition.
    Prints a one-line status. Silent if the repo is not present.
    """
    import subprocess as _sp
    if not _TT_FORGE_MODELS_PATH.exists():
        return
    try:
        result = _sp.run(
            ["git", "-C", str(_TT_FORGE_MODELS_PATH), "pull", "--ff-only", "-q"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  {GREEN}✓ tt-forge-models up to date{RESET}")
        else:
            print(f"  {YELLOW}⚠ tt-forge-models pull skipped: {result.stderr.strip()}{RESET}")
    except Exception as exc:
        print(f"  {YELLOW}⚠ tt-forge-models pull failed: {exc}{RESET}")


def _find_seed_requirements(model_id: str) -> pathlib.Path | None:
    """Return the requirements.txt Path for a seed model, or None if not found.

    Looks for ~/code/tt-forge-models/{prefix}/{backend}/requirements.txt where
    prefix is model_id.split("/")[0] and backend is model_id.split("/")[1] (if
    present). Falls back to prefix-only lookup for two-segment model IDs.

    Examples:
        "gliner/pytorch"           → ~/code/tt-forge-models/gliner/pytorch/requirements.txt
        "yolox/pytorch"            → ~/code/tt-forge-models/yolox/pytorch/requirements.txt
        "facebook/opt-125m"        → None (not a seed model path)
    """
    parts = model_id.split("/")
    candidates = []
    if len(parts) >= 2:
        candidates.append(_TT_FORGE_MODELS_PATH / parts[0] / parts[1] / "requirements.txt")
    candidates.append(_TT_FORGE_MODELS_PATH / parts[0] / "requirements.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
```

- [ ] **Step 4.4: Wire `_pull_tt_forge_models()` into `run_worker()` startup**

In `run_worker()`, the startup block is around line 1620. After the `_warm_hf_datasets(bestiary)` call, add:

```python
    _pull_tt_forge_models()
```

- [ ] **Step 4.5: Run tests**

```bash
python3 -m pytest tests/test_model_overlay.py -v 2>&1 | tail -15
```

Expected: all pass (skip is acceptable for tests that need tt-forge-models present).

- [ ] **Step 4.6: Commit**

```bash
git add lib/expedition/expedition_worker.py tests/test_model_overlay.py
git commit -m "feat: _pull_tt_forge_models() + _find_seed_requirements() for proactive dep loading"
```

---

## Task 5: Wire overlay into forge dispatch loop

**Files:**
- Modify: `lib/expedition/expedition_worker.py`

This is the main wiring task. Every model run in `run_worker()`'s dispatch loop gets an overlay. The overlay's python path is threaded through to `_compile_isolated()` via `item_dict`.

- [ ] **Step 5.1: Add `overlay_python` field to `_isolated_compile_worker` and re-exec logic**

`_isolated_compile_worker` is around line 1055. It receives `item_dict` as a plain dict. Add re-exec logic at the very top of `_isolated_compile_worker`, before anything else:

```python
def _isolated_compile_worker(item_dict: dict, chip_id: int, result_path: str) -> None:
    # If caller specified an overlay interpreter and we're not already running
    # under it, re-exec this process under the overlay python.  This is the
    # mechanism that makes the overlay's installed packages visible to the compile.
    _overlay_python = item_dict.pop("_overlay_python", None)
    if _overlay_python and _overlay_python != sys.executable:
        import os
        os.execv(_overlay_python, [_overlay_python] + sys.argv)
        # execv replaces this process — code below only runs in the new process
        # which will have _overlay_python == sys.executable.
    # ... rest of existing function unchanged ...
```

- [ ] **Step 5.2: Add `python` parameter to `_compile_isolated()`**

`_compile_isolated()` is at line 1138. Add an optional `python: pathlib.Path | None = None` parameter and pass it through `item_dict`:

```python
def _compile_isolated(item: "QueueItem", chip_id: int,
                      python: pathlib.Path | None = None) -> dict:
    item_dict = asdict(item)
    if python is not None:
        item_dict["_overlay_python"] = str(python)
    # ... rest of existing function unchanged ...
```

- [ ] **Step 5.3: Add `_FORGE_BASE_VENV` constant**

Near `_TT_FORGE_MODELS_PATH` (around line 659), add:

```python
_FORGE_BASE_VENV = pathlib.Path("/opt/ttforge-toolchain/venv")
```

- [ ] **Step 5.4: Wire overlay into the dispatch loop**

In `run_worker()`, the dispatch loop body is around line 1720. The overlay must wrap the entire model processing block. Find the line `start = time.time()` near the top of the loop body and add overlay creation just after it. Find the `bestiary.save()` call at the end of each loop iteration (around line 1959) and add overlay destruction after it.

The pattern to add:

```python
        # ── Per-model isolated overlay ────────────────────────────────────────
        from lib.expedition.model_overlay import create_overlay, destroy_overlay, install_requirements, _find_seed_requirements_from_overlay
        _overlay = create_overlay(item.model_id, base_venv=_FORGE_BASE_VENV)
        _overlay_installed: list[str] = []
        _req_file = _find_seed_requirements(item.model_id)
        if _req_file:
            _overlay_installed = install_requirements(_overlay, _req_file)
            if _overlay_installed:
                print(f"  {CYAN}↳ overlay: installed {_overlay_installed}{RESET}")
        try:
            # ... all existing model processing code stays inside this try ...
        finally:
            destroy_overlay(_overlay)
```

Pass `python=_overlay.python` to `_compile_isolated`:

```python
            cr = _compile_isolated(item, chip_id, python=_overlay.python)
```

Pass `pip_deps=_overlay_installed` to `bestiary.record_success()`:

```python
                bestiary.record_success(
                    ...,
                    pip_deps=_overlay_installed,
                )
```

Extract missing packages from `error_str` at the failure recording site (compile failure, around line 1953):

```python
            if not _item_is_jax:
                _missing = _extract_missing_package(error_str)
                bestiary.record_failure(item.model_id, run_number, error_str,
                                        env_fingerprint=_env_fp,
                                        missing_packages=[_missing] if _missing else [])
```

Add `_extract_missing_package()` helper before the dispatch loop (near `_try_install_missing`):

```python
def _extract_missing_package(error_str: str) -> str | None:
    """Extract package name from 'No module named X' error, or None."""
    import re
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_str)
    if not m:
        return None
    return m.group(1).split(".")[0]
```

- [ ] **Step 5.5: Verify the full test suite passes**

```bash
python3 -m pytest tests/test_model_overlay.py tests/test_bestiary_overlay.py tests/test_bestiary_envfix.py -v 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 5.6: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "feat: per-model overlay venv in forge dispatch loop — base env never mutated"
```

---

## Task 6: Wire overlay into XLA dispatch

**Files:**
- Modify: `lib/expedition/expedition_worker.py`

`_dispatch_xla_item()` is at line 1224. It currently uses `xla_python = Path.home() / "tt-xla" / "venv" / "bin" / "python3"` as the interpreter.

- [ ] **Step 6.1: Add `_XLA_BASE_VENV` constant**

Near `_FORGE_BASE_VENV`:

```python
_XLA_BASE_VENV = pathlib.Path.home() / "tt-xla" / "venv"
```

- [ ] **Step 6.2: Update `_dispatch_xla_item()` to accept and use an overlay python**

Add `overlay_python: pathlib.Path | None = None` parameter:

```python
def _dispatch_xla_item(
    item: "QueueItem",
    chip_id: int,
    run_number: int,
    bestiary_path: str,
    overlay_python: pathlib.Path | None = None,
) -> dict:
```

Inside the function, replace:

```python
    xla_python = Path.home() / "tt-xla" / "venv" / "bin" / "python3"
```

With:

```python
    xla_python = overlay_python if overlay_python is not None \
                 else Path.home() / "tt-xla" / "venv" / "bin" / "python3"
```

And update the check:

```python
    if not xla_python.exists():
        default["error_str"] = f"XLA python not found at {xla_python} — cannot compile JAX model"
        return default
```

- [ ] **Step 6.3: Create XLA overlay at the JAX dispatch call site**

In `run_worker()`, around line 1790 where `_item_is_jax` is checked:

```python
        if _item_is_jax:
            _print_progress_step(2, 3, "Routing to XLA worker (JAX)...")
            _xla_overlay = create_overlay(item.model_id, base_venv=_XLA_BASE_VENV)
            _xla_installed: list[str] = []
            _xla_req = _find_seed_requirements(item.model_id)
            if _xla_req:
                _xla_installed = install_requirements(_xla_overlay, _xla_req)
            try:
                cr = _dispatch_xla_item(item, chip_id, run_number, bestiary_path,
                                        overlay_python=_xla_overlay.python)
            finally:
                destroy_overlay(_xla_overlay)
            bestiary = Bestiary(path=bestiary_path)
```

- [ ] **Step 6.4: Verify tests pass**

```bash
python3 -m pytest tests/ --ignore=tests/lib -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6.5: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "feat: per-model overlay venv for XLA dispatch path"
```

---

## Task 7: Remove `_CUSTOM_DEP_MAP` as a reject gate

**Files:**
- Modify: `lib/expedition/expedition_worker.py`

With overlays proactively installing from tt-forge-models requirements.txt, the `_CUSTOM_DEP_MAP` gate in `_preflight_arch_check()` rejects models that should now compile. Remove the rejection — keep the map as a comment/reference.

- [ ] **Step 7.1: Find the gate in `_preflight_arch_check()`**

```bash
grep -n "_CUSTOM_DEP_MAP" lib/expedition/expedition_worker.py
```

The relevant block (around lines 748-758) looks like:

```python
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
                        f"MissingDependency: custom class {class_path!r} requires "
                        f"importable module '{module_root}' which is not installed")
```

- [ ] **Step 7.2: Replace with non-rejecting version**

Change the `_CUSTOM_DEP_MAP` check to a comment and let the `__import__` probe still run (so we catch genuinely unknown modules) but skip the map-based hard reject:

```python
        auto_map = cfg.get("auto_map", {})
        for _key, class_path in auto_map.items():
            if not isinstance(class_path, str) or "." not in class_path:
                continue
            module_root = class_path.split(".")[0]
            # _CUSTOM_DEP_MAP listed known deps previously rejected here.
            # With per-model overlay envs these packages are installed
            # proactively from tt-forge-models requirements.txt before
            # compile, so we no longer pre-reject them.
            try:
                __import__(module_root)
            except ImportError:
                return (True,
                        f"MissingDependency: custom class {class_path!r} requires "
                        f"importable module '{module_root}' which is not installed")
```

- [ ] **Step 7.3: Run tests**

```bash
python3 -m pytest tests/ --ignore=tests/lib -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 7.4: Commit**

```bash
git add lib/expedition/expedition_worker.py
git commit -m "refactor: remove _CUSTOM_DEP_MAP reject gate — overlays handle these deps now"
```

---

## Task 8: `scripts/missing_deps_report.py`

**Files:**
- Create: `scripts/missing_deps_report.py`

- [ ] **Step 8.1: Write the script**

```python
#!/usr/bin/env python3
"""Report which Python packages are blocking the most models.

Usage:
    python3 scripts/missing_deps_report.py
    python3 scripts/missing_deps_report.py --json
    python3 scripts/missing_deps_report.py --bestiary path/to/bestiary.json

Reads data/bestiary.json (or --bestiary path) and prints a ranked table of
packages that appear in failed entries' missing_packages field.
"""
import argparse, json, pathlib, sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bestiary",
        default="data/bestiary.json",
        help="Path to bestiary.json (default: data/bestiary.json)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output machine-readable JSON instead of a table",
    )
    args = parser.parse_args()

    bestiary_path = pathlib.Path(args.bestiary)
    if not bestiary_path.exists():
        print(f"Error: {bestiary_path} not found", file=sys.stderr)
        sys.exit(1)

    # Use Bestiary class for consistency.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lib.expedition.bestiary import Bestiary
    b = Bestiary(path=bestiary_path)
    report = b.missing_dep_report()

    if not report:
        print("No missing_packages recorded in bestiary yet.")
        return

    if args.as_json:
        print(json.dumps(report, indent=2))
        return

    col1, col2, col3 = 24, 16, 50
    header = f"{'Package':<{col1}}  {'Models blocked':<{col2}}  {'Example models'}"
    print(header)
    print("─" * (col1 + col2 + col3 + 4))
    for row in report:
        examples = ", ".join(row["models"][:3])
        if len(row["models"]) > 3:
            examples += f", +{len(row['models']) - 3} more"
        print(f"{row['package']:<{col1}}  {row['count']:<{col2}}  {examples}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Make it executable and test it**

```bash
chmod +x scripts/missing_deps_report.py
python3 scripts/missing_deps_report.py
```

Expected: either "No missing_packages recorded yet" or a ranked table (depending on current bestiary state).

- [ ] **Step 8.3: Commit**

```bash
git add scripts/missing_deps_report.py
git commit -m "feat: scripts/missing_deps_report.py — ranked table of packages blocking models"
```

---

## Task 9: `scripts/clean_bestiary.py`

**Files:**
- Create: `scripts/clean_bestiary.py`

- [ ] **Step 9.1: Write the script**

```python
#!/usr/bin/env python3
"""One-shot cleanup of stale harness-caused bestiary entries.

Run once after merging harness-hardening-envfix to evict entries that failed
for infrastructure reasons rather than forge/XLA limitations.

Clears:
  - Entries whose last_error contains "cats_image.jpeg"
  - Entries whose error_category == "wrong_backend"
  - Entries whose env_fingerprint differs from the current env on version-signal errors

Usage:
    python3 scripts/clean_bestiary.py           # live run
    python3 scripts/clean_bestiary.py --dry-run # preview only
    python3 scripts/clean_bestiary.py --bestiary path/to/bestiary.json
"""
import argparse, pathlib, sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be cleared without modifying the file")
    parser.add_argument("--bestiary", default="data/bestiary.json",
                        help="Path to bestiary.json (default: data/bestiary.json)")
    args = parser.parse_args()

    bestiary_path = pathlib.Path(args.bestiary)
    if not bestiary_path.exists():
        print(f"Error: {bestiary_path} not found", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from lib.expedition.bestiary import Bestiary, _current_env_fingerprint

    b = Bestiary(path=bestiary_path)
    current_fp = _current_env_fingerprint()

    cats_cleared = b.clear_entries_matching(error_contains="cats_image.jpeg")
    wrong_backend_cleared = [
        mid for mid, entry in list(b.failed.items())
        if entry.get("error_category") == "wrong_backend"
    ]
    if not args.dry_run:
        for mid in wrong_backend_cleared:
            del b._data["failed"][mid]
    env_cleared = b.clear_stale_env_failures(current_fp)

    total = len(cats_cleared) + len(wrong_backend_cleared) + len(env_cleared)

    print(f"cats_image.jpeg entries:  {len(cats_cleared)}")
    for mid in cats_cleared:
        print(f"  - {mid}")
    print(f"wrong_backend entries:    {len(wrong_backend_cleared)}")
    for mid in wrong_backend_cleared:
        print(f"  - {mid}")
    print(f"stale env entries:        {len(env_cleared)}")
    for mid in env_cleared:
        print(f"  - {mid}")
    print(f"\nTotal: {total} entries {'would be' if args.dry_run else ''} cleared")

    if not args.dry_run and total > 0:
        b.save()
        print("bestiary.json updated.")
    elif args.dry_run:
        print("(dry-run — no changes written)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Test dry-run mode**

```bash
chmod +x scripts/clean_bestiary.py
python3 scripts/clean_bestiary.py --dry-run
```

Expected: prints counts for each category (may be non-zero if harness-hardening-envfix hasn't been run yet) with "(dry-run — no changes written)" at the end.

- [ ] **Step 9.3: Commit**

```bash
git add scripts/clean_bestiary.py
git commit -m "feat: scripts/clean_bestiary.py — one-shot stale entry eviction"
```

---

## Task 10: `docs/overlay-deps.md` + `CLAUDE.md`

**Files:**
- Create: `docs/overlay-deps.md`
- Modify: `CLAUDE.md`

- [ ] **Step 10.1: Write `docs/overlay-deps.md`**

```markdown
# Overlay Deps — Per-Model Isolated Environments

## What this is

Every model compile runs inside a thin disposable Python venv "overlay" layered on top of the base forge or XLA venv. The overlay inherits all base packages via symlinks (`--system-site-packages --symlinks`) but any pip installs go into the overlay only. The overlay is deleted after the model finishes.

This means:
- A bad install can't corrupt subsequent models in the same run
- The base forge/XLA env stays pristine across expeditions
- Each model gets a clean slate on every run

## How overlays work

1. `create_overlay(model_id, base_venv)` creates `/tmp/compiletron-overlay-{model_id}/` in ~10ms
2. For seed models: `_find_seed_requirements(model_id)` finds `~/code/tt-forge-models/{prefix}/{backend}/requirements.txt`
3. `install_requirements(overlay, req_file)` pip-installs the safe subset into the overlay
4. The compile subprocess is launched under `overlay/bin/python3` via the re-exec mechanism
5. On finish: `destroy_overlay(overlay)` removes the directory

## Adding a dependency for a seed model

If a seed model needs a pip package, add it to its `requirements.txt` in `tt-forge-models`:

```
~/code/tt-forge-models/{model_name}/{backend}/requirements.txt
```

Example — `gliner/pytorch/requirements.txt`:
```
gliner
```

The next expedition run will `git pull` tt-forge-models and pick up the new dep automatically.

## Requirements.txt parsing rules

The parser skips unsafe lines to prevent breaking the base env:
- Lines starting with `--` (index URLs, extra flags)
- Lines starting with `-r` (recursive includes)
- Lines containing `://` (git+https VCS deps)
- Lines whose package name matches a protected core package: `torch`, `transformers`, `forge`, `jax`, `jaxlib`, `flax`, `ttnn`, `tt_lib`, `torchvision`, `torchaudio`

Everything else is passed to `pip install`.

## Tracking what got installed

The bestiary records installed deps on compiled entries:
```json
"gliner/pytorch": {
  "pip_deps": ["gliner"],
  ...
}
```

And missing packages on failed entries:
```json
"surya/pytorch": {
  "missing_packages": ["surya-ocr"],
  ...
}
```

## Viewing the missing deps report

```bash
python3 scripts/missing_deps_report.py
```

Shows a ranked table of packages blocking the most models. Add `--json` for machine-readable output.

## One-shot stale entry cleanup

After merging harness-hardening-envfix, run:

```bash
python3 scripts/clean_bestiary.py --dry-run   # preview
python3 scripts/clean_bestiary.py              # apply
```

This evicts ~60 entries that failed for harness/env reasons rather than forge/XLA limitations.
```

- [ ] **Step 10.2: Append to `CLAUDE.md`**

Append at the end of `CLAUDE.md`:

```markdown
## Isolated Overlay Deps (2026-05-27)

**Goal:** Base forge/XLA env never mutated during model runs.

**How it works:** Each model gets a `~10ms` disposable venv overlay (`--system-site-packages --symlinks`). Seed model deps pre-installed from `~/code/tt-forge-models/{model}/requirements.txt` before compile. Overlay destroyed after the model finishes.

**Key files:** `lib/expedition/model_overlay.py`, `docs/overlay-deps.md`

**New scripts:**
- `python3 scripts/missing_deps_report.py` — ranked table of packages blocking models
- `python3 scripts/clean_bestiary.py --dry-run` — preview stale entry cleanup
- `python3 scripts/clean_bestiary.py` — apply stale entry cleanup (run once after harness-hardening-envfix merge)

**Adding a dep for a seed model:** Edit `~/code/tt-forge-models/{model}/{backend}/requirements.txt`. Picked up automatically on next expedition via startup `git pull`.
```

- [ ] **Step 10.3: Commit**

```bash
git add docs/overlay-deps.md CLAUDE.md
git commit -m "docs: overlay-deps guide + CLAUDE.md update"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `ModelOverlay` dataclass + `create_overlay` + `destroy_overlay` | Task 1 ✓ |
| `_parse_requirements` + `install_requirements` | Task 2 ✓ |
| `pip_deps` on `record_success()` | Task 3 ✓ |
| `missing_packages` on `record_failure()` | Task 3 ✓ |
| `missing_dep_report()` method | Task 3 ✓ |
| `_pull_tt_forge_models()` | Task 4 ✓ |
| `_find_seed_requirements()` | Task 4 ✓ |
| Overlay wired into forge dispatch loop | Task 5 ✓ |
| `_compile_isolated` overlay python parameter | Task 5 ✓ |
| `_extract_missing_package()` helper | Task 5 ✓ |
| Overlay wired into XLA dispatch | Task 6 ✓ |
| `_CUSTOM_DEP_MAP` gate removed | Task 7 ✓ |
| `scripts/missing_deps_report.py` | Task 8 ✓ |
| `scripts/clean_bestiary.py` | Task 9 ✓ |
| `docs/overlay-deps.md` | Task 10 ✓ |
| `CLAUDE.md` updated | Task 10 ✓ |
| `_FORGE_BASE_VENV` + `_XLA_BASE_VENV` constants | Task 5+6 ✓ |
| `git pull tt-forge-models` at startup | Task 4 ✓ |

**Placeholder scan:** No TBDs. All code blocks complete. ✓

**Type consistency:**
- `install_requirements(overlay: ModelOverlay, req_file: Path) -> list[str]` — defined Task 2, called Task 5 ✓
- `create_overlay(model_id: str, base_venv: Path, overlay_root: Path | None = None) -> ModelOverlay` — defined Task 1, called Task 5+6 ✓
- `destroy_overlay(overlay: ModelOverlay) -> None` — defined Task 1, called Task 5+6 ✓
- `_find_seed_requirements(model_id: str) -> Path | None` — defined Task 4, called Task 5+6 ✓
- `record_success(..., pip_deps: list[str] = ()) -> None` — defined Task 3, called Task 5 ✓
- `record_failure(..., missing_packages: list[str] = ()) -> None` — defined Task 3, called Task 5 ✓
- `missing_dep_report() -> list[dict]` — defined Task 3, used Task 8 ✓
