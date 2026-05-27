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
