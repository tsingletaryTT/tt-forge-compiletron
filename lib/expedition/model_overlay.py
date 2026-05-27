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
