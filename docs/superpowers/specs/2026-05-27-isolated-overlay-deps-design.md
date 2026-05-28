# Isolated Overlay Envs + Dependency Tracking

**Date:** 2026-05-27
**Branch:** isolated-overlay-deps (to be created)
**Goal:** Give every model an isolated Python env overlay so dep installs never corrupt the base forge/XLA env, while tracking which packages are missing across the fleet.

---

## Problem Summary

Three related problems:

1. **`_try_install_missing` mutates the base env** — reactive pip installs go directly into `/opt/ttforge-toolchain/venv/` or `~/tt-xla/venv/`. A bad install (wrong version, conflicting pin) can corrupt subsequent models in the same run.

2. **Per-model deps in tt-forge-models are ignored** — `~/code/tt-forge-models/{prefix}/{backend}/requirements.txt` exists for 27+ seed models and is the ground-truth validated dep list. The harness never reads it.

3. **No visibility into what's blocking models** — when a model fails with `No module named 'surya'`, that fact is buried in free-text `last_error`. There's no aggregated view of "package X is blocking N models."

---

## Solution Overview

**Isolated overlay venv per model**: Create a thin symlinked venv (10ms, `--system-site-packages --symlinks`) before each model runs. Installs go into the overlay only. Subprocess is launched under the overlay's interpreter. Overlay is deleted after the model finishes (success or fail).

**Proactive install from tt-forge-models**: For seed models, read `requirements.txt` before the model loads and install into the overlay. No `No module named` surprises.

**Reactive install still works**: For frontier models (no requirements.txt), `_try_install_missing` still fires on `No module named` errors — but now installs into the overlay, not the base env.

**Tracking**: Record installed packages and missing packages in bestiary entries. A standalone report script aggregates missing packages across all failed entries.

**tt-forge-models git pull on startup**: Keep requirements.txt files current without manual intervention.

**Clean script**: One-shot `scripts/clean_bestiary.py` to evict the ~60 stale harness-caused entries accumulated before this fix.

---

## Architecture

### New module: `lib/expedition/model_overlay.py`

Single responsibility: create, populate, and destroy a per-model venv overlay.

```python
@dataclass
class ModelOverlay:
    path: Path          # e.g. /tmp/compiletron-overlay-gliner_pytorch_abc123
    python: Path        # path/bin/python3
    base_venv: Path     # the venv this overlays

def create_overlay(model_id: str, base_venv: Path) -> ModelOverlay: ...
def install_requirements(overlay: ModelOverlay, req_file: Path) -> list[str]: ...
    # Returns list of package names actually installed.
    # Skips: git+https://, --extra-index-url, -r includes, blank lines, comments.
    # Uses: overlay.python -m pip install -q --no-deps <pkg> for each safe line.
    # Falls back to pip install -r for the filtered safe subset.
def destroy_overlay(overlay: ModelOverlay) -> None: ...
    # shutil.rmtree(overlay.path), fails silently.
```

### Changes to `lib/expedition/expedition_worker.py`

**`_pull_tt_forge_models()`** — new function called once at `run_worker()` startup:
```python
def _pull_tt_forge_models() -> None:
    # git pull --ff-only in ~/code/tt-forge-models
    # Prints: "✓ tt-forge-models updated" or "⚠ git pull skipped: <reason>"
    # Fails open — stale tree is better than blocked expedition.
```

**`_find_seed_requirements(item: QueueItem) -> Path | None`** — new function:
```python
# Returns ~/code/tt-forge-models/{prefix}/{backend}/requirements.txt if it exists.
# prefix = item.model_id.split("/")[0]
# backend = item.model_id.split("/")[-1] if not frontier else None
```

**`run_worker()` startup block** — add after existing warmup calls:
```python
_pull_tt_forge_models()
```

**`run_worker()` dispatch loop** — wrap each model in an overlay:
```python
overlay = create_overlay(item.model_id, base_venv=_FORGE_BASE_VENV)
try:
    req_file = _find_seed_requirements(item)
    installed_pkgs: list[str] = []
    if req_file:
        installed_pkgs = install_requirements(overlay, req_file)
    cr = _compile_isolated(item, chip_id, python=overlay.python)
    # ... existing result handling ...
    # On success: pass installed_pkgs to record_success()
    # On failure: extract missing packages, pass to record_failure()
finally:
    destroy_overlay(overlay)
```

**`_compile_isolated(item, chip_id, python=None)`** — add optional `python` parameter:
```python
# If python is provided, use it instead of sys.executable for the subprocess.
# Forge subprocess currently uses multiprocessing.Process (inherits interpreter).
# Change: use subprocess.Popen([str(python), "-c", ...]) instead, or pass
# python path through to the isolated worker via item_dict.
```

**`_dispatch_xla_item()`** — same overlay approach, using `~/tt-xla/venv` as base:
```python
overlay = create_overlay(item.model_id, base_venv=Path.home() / "tt-xla" / "venv")
# xla_python becomes overlay.python instead of the hardcoded xla_python path
```

**`_try_install_missing()`** — keep as-is but it now installs into the overlay automatically (since the subprocess runs under the overlay interpreter and `sys.executable` inside the subprocess points to the overlay's python).

**Remove**: the early-return in `_preflight_arch_check()` that returns `(True, "MissingDependency...")` when a module root is found in `_CUSTOM_DEP_MAP`. With overlays + proactive install from tt-forge-models requirements.txt, these models should get a compile attempt rather than being pre-emptively rejected. Keep `_CUSTOM_DEP_MAP` as documentation of known deps, but stop using it as a gate. Its entries are candidates for tt-forge-models `requirements.txt` files if not already there.

**Base venv constant**:
```python
_FORGE_BASE_VENV = Path("/opt/ttforge-toolchain/venv")
_XLA_BASE_VENV   = Path.home() / "tt-xla" / "venv"
```

### Changes to `lib/expedition/bestiary.py`

**`record_success()`** — add optional `pip_deps: list[str] = ()` kwarg:
```python
# Stored as: compiled[model_id]["pip_deps"] = ["gliner", "FlagEmbedding"]
# Only written when non-empty.
```

**`record_failure()`** — add optional `missing_packages: list[str] = ()` kwarg:
```python
# Stored as: failed[model_id]["missing_packages"] = ["surya-ocr"]
# Merged with existing list (union) — accumulates across runs.
```

**`missing_dep_report() -> list[dict]`** — new method:
```python
# Returns list of {"package": str, "count": int, "models": list[str]}
# sorted descending by count.
# Reads missing_packages from all failed entries.
```

### New script: `scripts/missing_deps_report.py`

```
Usage: python3 scripts/missing_deps_report.py [--json]

Package              Models blocked    Example models
────────────────────────────────────────────────────
surya-ocr            8                 suryaocr/pytorch, ...
torchaudio           4                 seamless_m4t/pytorch, minicpm_o_2_6/pytorch
gliner               2                 gliner/pytorch, ...
```

Reads `data/bestiary.json` and calls `bestiary.missing_dep_report()`. Optional `--json` flag for machine-readable output.

### New script: `scripts/clean_bestiary.py`

One-shot cleanup of accumulated stale entries. Run once after merging `harness-hardening-envfix`:

```
Usage: python3 scripts/clean_bestiary.py [--dry-run]

Clears:
  - Entries where last_error contains "cats_image.jpeg"
  - Entries where error_category == "wrong_backend"  
  - Entries where env_fingerprint differs from current env on version-signal errors

Prints summary of what was (or would be) cleared.
```

### New doc: `docs/overlay-deps.md`

Documents:
- How the overlay lifecycle works and why
- How to add a new dep to a seed model (edit `~/code/tt-forge-models/{model}/pytorch/requirements.txt`)
- How to read the missing deps report
- What the `pip_deps` and `missing_packages` bestiary fields mean
- How to run the clean script

---

## Requirements.txt Parsing Rules

Safe to install (process the line):
- Plain package name: `gliner`, `FlagEmbedding`
- Package with version spec: `surya-ocr==0.15.4`, `omegaconf>=2.3.0`
- Package with extras: `some-pkg[extra]`

Skip silently (log at DEBUG):
- Lines starting with `--` (index URLs, flags)
- Lines starting with `-r` (recursive includes)
- Lines starting with `git+` or containing `://`
- Lines containing `torch`, `transformers`, `forge` (protect core deps)
- Blank lines, comment lines (`#`)

---

## Overlay Subprocess Wiring

`_compile_isolated` currently uses `multiprocessing.Process` which inherits the parent interpreter. To use the overlay interpreter it must switch to `subprocess.Popen`:

```python
proc = subprocess.Popen(
    [str(overlay_python), "-c",
     "import pickle,sys; fn,args=pickle.loads(sys.stdin.buffer.read()); fn(*args)"],
    stdin=subprocess.PIPE, ...
)
```

Or simpler: pass `overlay_python` as an env var into the existing `multiprocessing.Process` and have `_isolated_compile_worker` re-exec itself under that interpreter if it differs from `sys.executable`. This avoids the IPC refactor.

**Recommended**: pass `overlay_python` path into `item_dict` (already serialized as JSON for the subprocess). The isolated worker reads it and if it differs from `sys.executable`, re-execs:
```python
if overlay_python and Path(overlay_python) != Path(sys.executable):
    os.execv(overlay_python, [overlay_python] + sys.argv)
```

This is the minimal-change path — no IPC refactor needed.

---

## Files Changed

| File | Change |
|---|---|
| `lib/expedition/model_overlay.py` | New — overlay create/install/destroy |
| `lib/expedition/expedition_worker.py` | `_pull_tt_forge_models()`, `_find_seed_requirements()`, overlay wiring in dispatch loop, `_compile_isolated` python param, `_dispatch_xla_item` overlay, remove `_CUSTOM_DEP_MAP` preflight rejection |
| `lib/expedition/bestiary.py` | `pip_deps` on `record_success()`, `missing_packages` on `record_failure()`, `missing_dep_report()` method |
| `scripts/missing_deps_report.py` | New — ranked table of blocking packages |
| `scripts/clean_bestiary.py` | New — one-shot stale entry cleanup |
| `docs/overlay-deps.md` | New — operator guide |
| `CLAUDE.md` | Updated with overlay startup sequence and new scripts |
| `tests/test_model_overlay.py` | New — overlay lifecycle + requirements parsing tests |
| `tests/test_bestiary_overlay.py` | New — pip_deps/missing_packages field tests + report format |

---

## Success Criteria

1. Running a seed model with a `requirements.txt` installs its deps into `/tmp/compiletron-overlay-*` — the base forge venv is unchanged after the run.
2. A frontier model that hits `No module named 'X'` installs `X` into its overlay only — visible in `pip_deps` on the compiled entry or `missing_packages` on the failed entry.
3. `python3 scripts/missing_deps_report.py` prints a ranked table from `data/bestiary.json`.
4. `python3 scripts/clean_bestiary.py --dry-run` shows the ~60 stale entries without modifying the file.
5. `/tmp` is clean after every model run — no orphaned overlay dirs.
6. `docs/overlay-deps.md` exists and explains how to add a dep to a seed model.
7. A model previously rejected by `_CUSTOM_DEP_MAP` (e.g. `mamba-ssm`) now reaches the compile step when its overlay has the dep installed.
