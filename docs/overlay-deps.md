# Overlay Deps — Per-Model Isolated Environments

## What this is

Every model compile runs inside a thin disposable Python venv "overlay" layered on top of the base forge or XLA venv. The overlay inherits all base packages via symlinks (`--system-site-packages --symlinks`) but any pip installs go into the overlay only. The overlay is deleted after the model finishes.

This means:
- A bad install can't corrupt subsequent models in the same run
- The base forge/XLA env stays pristine across expeditions
- Each model gets a clean slate on every run

## How overlays work

1. `create_overlay(model_id, base_venv)` creates `/tmp/compiletron-overlay-{model_id}_{hash}/` in ~10ms
2. For seed models: `_find_seed_requirements(model_id)` finds `~/code/tt-forge-models/{prefix}/{backend}/requirements.txt`
3. `install_requirements(overlay, req_file)` pip-installs the safe subset into the overlay
4. The compile subprocess is launched under `overlay/bin/python3` via the re-exec mechanism
5. On finish (success or fail): `destroy_overlay(overlay)` removes the directory

## Adding a dependency for a seed model

If a seed model needs a pip package, add it to its `requirements.txt` in `tt-forge-models`:

```
~/code/tt-forge-models/{model_name}/{backend}/requirements.txt
```

Example — `gliner/pytorch/requirements.txt`:
```
gliner
```

The next expedition run will `git pull` tt-forge-models and pick up the new dep automatically (via `_pull_tt_forge_models()` at startup).

## Requirements.txt parsing rules

The parser skips unsafe lines to prevent breaking the base env:

| Pattern | Skipped |
|---|---|
| Lines starting with `--` | Index URLs, extra flags |
| Lines starting with `-r` | Recursive includes |
| Lines containing `://` | `git+https` VCS deps |
| Protected packages | `torch`, `transformers`, `forge`, `jax`, `jaxlib`, `flax`, `ttnn`, `tt_lib`, `torchvision`, `torchaudio` |
| Blank lines and comments | — |

Everything else is passed to `pip install -q --no-build-isolation`.

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

`missing_packages` accumulates across runs (set-union) so a model that fails twice with different missing packages shows both.

## Viewing the missing deps report

```bash
python3 scripts/missing_deps_report.py
```

Shows a ranked table of packages blocking the most models:

```
Package                   Models blocked    Example models
────────────────────────────────────────────────────────────────────────────────
surya-ocr                 8                 suryaocr/pytorch, ...
torchaudio                4                 seamless_m4t/pytorch, ...
gliner                    2                 gliner/pytorch, ...
```

Add `--json` for machine-readable output.

## One-shot stale entry cleanup

After merging `harness-hardening-envfix`, run:

```bash
python3 scripts/clean_bestiary.py --dry-run   # preview (~44 entries on the current bestiary)
python3 scripts/clean_bestiary.py              # apply
```

This evicts entries that failed for harness/env reasons:
- `cats_image.jpeg` errors (dataset cache miss — fixed by `_warm_hf_datasets`)
- `wrong_backend` errors (JAX models dispatched to forge — fixed by proper routing)
- Stale env-version failures (older env pinned in `env_fingerprint` — fixed by env upgrade)

## Overlay lifecycle in code

The overlay is created in `run_worker()` before each model and destroyed in a `finally` block:

```python
_overlay = create_overlay(item.model_id, base_venv=_FORGE_BASE_VENV)
_overlay_installed = []
try:
    req = _find_seed_requirements(item.model_id)
    if req:
        _overlay_installed = install_requirements(_overlay, req)
    cr = _compile_isolated(item, chip_id, python=_overlay.python)
    ...
finally:
    destroy_overlay(_overlay)
```

The overlay python path is passed through `item_dict["_overlay_python"]` to `_isolated_compile_worker`, which re-execs itself under the overlay interpreter via `os.execv` if the path differs from `sys.executable`.
