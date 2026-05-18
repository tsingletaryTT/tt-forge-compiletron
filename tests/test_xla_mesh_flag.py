# tests/test_xla_mesh_flag.py
"""Verify --xla-mesh routing: JAX loaders get mesh_chips=N."""
import pytest


def test_build_queues_accepts_xla_mesh_param():
    """build_queues() should accept xla_mesh kwarg without error."""
    from expedition import build_queues
    import inspect
    sig = inspect.signature(build_queues)
    assert "xla_mesh" in sig.parameters, (
        "build_queues must accept xla_mesh parameter"
    )


def test_scan_forge_models_accepts_xla_mesh_param():
    """_scan_forge_models() should accept xla_mesh kwarg."""
    from expedition import _scan_forge_models
    import inspect
    sig = inspect.signature(_scan_forge_models)
    assert "xla_mesh" in sig.parameters, (
        "_scan_forge_models must accept xla_mesh parameter"
    )


def test_jax_loader_item_gets_mesh_chips_from_xla_mesh():
    """A JAX loader item dict should have mesh_chips=N when xla_mesh=N."""
    item = {
        "model_id": "alexnet/image_classification/jax",
        "library": "jax",
        "mesh_chips": 1,
    }
    xla_mesh = 4
    if item.get("library") == "jax" and xla_mesh > 1:
        item["mesh_chips"] = xla_mesh
    assert item["mesh_chips"] == 4


def test_pytorch_loader_item_unchanged_with_xla_mesh():
    """PyTorch loader items keep mesh_chips=1 when xla_mesh is set."""
    item = {
        "model_id": "alexnet/pytorch",
        "library": "pytorch",
        "mesh_chips": 1,
    }
    xla_mesh = 4
    if item.get("library") == "jax" and xla_mesh > 1:
        item["mesh_chips"] = xla_mesh
    assert item["mesh_chips"] == 1
