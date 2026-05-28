# tests/test_model_overlay.py
import pathlib, pytest
from lib.expedition.model_overlay import (
    create_overlay, destroy_overlay, ModelOverlay,
    _parse_requirements, install_requirements,
)


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
    try:
        assert str(overlay.path).startswith("/tmp/")
    finally:
        # Ensure the overlay is cleaned up even if the assertion above fails.
        destroy_overlay(overlay)


def test_destroy_overlay_is_idempotent(tmp_path):
    """destroy_overlay on a non-existent path does not raise."""
    overlay = ModelOverlay(
        path=tmp_path / "nonexistent",
        python=tmp_path / "nonexistent" / "bin" / "python3",
        base_venv=pathlib.Path("/opt/ttforge-toolchain/venv"),
    )
    destroy_overlay(overlay)  # must not raise


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
        "torch==2.5.1\n"
        "transformers>=4.0\n"
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
    import lib.expedition.model_overlay as _mod
    calls = []
    # Patch _sp (the module-level `import subprocess as _sp` alias) so the
    # monkeypatch intercepts the call regardless of how subprocess is imported
    # elsewhere.
    monkeypatch.setattr(_mod._sp, "run", lambda cmd, **kw: calls.append(cmd))

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
    assert len(calls) == 1  # single pip install call
    assert str(overlay.python) in calls[0]


# ── _find_seed_requirements ──────────────────────────────────────────────────

from lib.expedition.expedition_worker import _find_seed_requirements


def test_find_seed_requirements_returns_path_when_exists():
    """Returns path for a seed model that has requirements.txt in tt-forge-models."""
    result = _find_seed_requirements("gliner/pytorch")
    if result is None:
        pytest.skip("tt-forge-models not present or gliner/pytorch has no requirements.txt")
    assert result.exists()
    assert result.name == "requirements.txt"


def test_find_seed_requirements_returns_none_for_unknown():
    """Returns None when no requirements.txt exists for the model."""
    result = _find_seed_requirements("nonexistent/model")
    assert result is None


def test_find_seed_requirements_returns_none_for_frontier():
    """A HuggingFace model ID with no matching tt-forge-models directory returns None."""
    result = _find_seed_requirements("facebook/opt-125m")
    assert result is None
