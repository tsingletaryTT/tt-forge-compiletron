# lib/expedition/model_overlay.py
"""Thin per-model venv overlay — isolates pip installs from the base forge/XLA env.

Each model gets a symlinked venv (~10ms to create) that inherits the base env's
packages via --system-site-packages. Installs go into the overlay only. The overlay
is destroyed after the model finishes (success or fail). The base env is never touched.
"""
from __future__ import annotations
import hashlib
import shutil
import subprocess as _sp
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
    # 8-char sha1 suffix prevents collision when model_id values differ only by
    # separator (e.g. "org/model-name" vs "org_model-name" both become the same
    # slug without the hash).
    safe_name = (
        model_id.replace("/", "_").replace(" ", "-")
        + "_"
        + hashlib.sha1(model_id.encode()).hexdigest()[:8]
    )
    if overlay_root is None:
        overlay_root = Path(tempfile.gettempdir())
    overlay_path = overlay_root / f"compiletron-overlay-{safe_name}"
    # Remove any stale overlay from a previous crashed run.
    if overlay_path.exists():
        shutil.rmtree(overlay_path, ignore_errors=True)
    # Root the new venv to base_venv's interpreter so it inherits exactly the
    # forge/XLA packages.  Using subprocess avoids venv.create() which always
    # roots to the *currently running* Python, ignoring base_venv entirely.
    try:
        _sp.run(
            [
                str(base_venv / "bin" / "python3"),
                "-m", "venv",
                str(overlay_path),
                "--system-site-packages",
                "--symlinks",
            ],
            check=True,
            capture_output=True,
        )
    except Exception:
        # Clean up any partial directory so the next attempt starts fresh.
        shutil.rmtree(overlay_path, ignore_errors=True)
        raise
    python = overlay_path / "bin" / "python3"
    return ModelOverlay(path=overlay_path, python=python, base_venv=base_venv)


def destroy_overlay(overlay: ModelOverlay) -> None:
    """Delete the overlay directory. Silent if already gone."""
    shutil.rmtree(overlay.path, ignore_errors=True)
