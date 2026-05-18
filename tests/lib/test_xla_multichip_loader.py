# tests/lib/test_xla_multichip_loader.py
"""Unit tests for XLA multi-chip routing — no hardware required."""
import sys
import types
import pytest
import unittest.mock as mock
from dataclasses import dataclass


def test_loader_meta_defaults():
    """_LoaderMeta defaults: use_shard_map=False, axis_name='batch'."""
    from lib.expedition.expedition_worker_xla import _LoaderMeta
    meta = _LoaderMeta()
    assert meta.use_shard_map is False
    assert meta.axis_name == "batch"


def test_loader_meta_type_a_fields():
    """_LoaderMeta for Type A: use_shard_map=True, axis_name='X'."""
    from lib.expedition.expedition_worker_xla import _LoaderMeta
    meta = _LoaderMeta(use_shard_map=True, axis_name="X")
    assert meta.use_shard_map is True
    assert meta.axis_name == "X"


def test_build_loader_xla_returns_tuple_for_frontier():
    """_build_loader_xla must return (callable, _LoaderMeta) for frontier models."""
    from lib.expedition.expedition_worker_xla import _build_loader_xla, _LoaderMeta, QueueItem

    item = QueueItem(
        model_id="org/gpt2-large",
        display_name="GPT2 Large",
        task="text-generation",
        source="huggingface",
        rarity="uncommon",
        hf_downloads=100000,
        hf_created_at=None,
        mesh_chips=1,
        loader_module=None,
        loader_class=None,
        is_frontier=True,
    )

    result = _build_loader_xla(item)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"_build_loader_xla must return (loader, meta) tuple, got {type(result)}"
    )
    loader, meta = result
    assert callable(loader), "First element must be callable"
    assert isinstance(meta, _LoaderMeta), "Second element must be _LoaderMeta"
    assert meta.use_shard_map is False, "Frontier models use data-parallel (no shard_map)"


def _make_seed_item(mesh_chips=1, model_id="fake/model/jax",
                    loader_module="_forgems.fake.loader", loader_class="ModelLoader"):
    from lib.expedition.expedition_worker_xla import QueueItem
    return QueueItem(
        model_id=model_id,
        display_name="Fake",
        task="image-classification",
        source="custom",
        rarity="familiar",
        hf_downloads=None,
        hf_created_at=None,
        mesh_chips=mesh_chips,
        loader_module=loader_module,
        loader_class=loader_class,
        is_frontier=False,
    )


def _call_build_loader_with_mock_instance(item, instance):
    """Patch importlib so _build_loader_xla uses our mock instance."""
    import lib.expedition.expedition_worker_xla as worker

    fake_mod = types.ModuleType("_forgems.fake.loader")
    fake_mod.ModelLoader = mock.MagicMock(return_value=instance)
    fake_mod.ModelVariant = None

    with mock.patch.dict(sys.modules, {"_forgems.fake.loader": fake_mod}):
        with mock.patch("importlib.import_module", return_value=fake_mod):
            return worker._build_loader_xla(item)


def test_seed_single_chip_returns_default_meta():
    """Seed model with mesh_chips=1 → _LoaderMeta(use_shard_map=False, axis_name='batch')."""
    from lib.expedition.expedition_worker_xla import _LoaderMeta

    instance = mock.MagicMock(spec=[])  # no attributes, simulates Type B
    instance.load_model = mock.MagicMock(return_value=mock.MagicMock())
    instance.load_inputs = mock.MagicMock(return_value=mock.MagicMock())

    item = _make_seed_item(mesh_chips=1)
    loader, meta = _call_build_loader_with_mock_instance(item, instance)

    assert callable(loader)
    assert isinstance(meta, _LoaderMeta)
    assert meta.use_shard_map is False
    assert meta.axis_name == "batch"


def test_seed_multichip_type_b_uses_data_parallel_meta():
    """Type B (no load_multichip_model, mesh_chips>1) → use_shard_map=False, axis='batch'."""
    from lib.expedition.expedition_worker_xla import _LoaderMeta

    instance = mock.MagicMock(spec=[])  # no load_multichip_model
    instance.load_model = mock.MagicMock(return_value=mock.MagicMock())

    item = _make_seed_item(mesh_chips=4)
    loader, meta = _call_build_loader_with_mock_instance(item, instance)

    assert meta.use_shard_map is False
    assert meta.axis_name == "batch"


def test_seed_multichip_type_a_uses_shard_map_meta():
    """Type A (has load_multichip_model, mesh_chips>1) → use_shard_map=True, axis='X'."""
    from lib.expedition.expedition_worker_xla import _LoaderMeta

    instance = mock.MagicMock()  # has load_multichip_model
    instance.load_model = mock.MagicMock(return_value=mock.MagicMock())
    instance.load_multichip_model = mock.MagicMock(return_value=mock.MagicMock())

    item = _make_seed_item(mesh_chips=4)
    loader, meta = _call_build_loader_with_mock_instance(item, instance)

    assert meta.use_shard_map is True
    assert meta.axis_name == "X"


def test_seed_type_a_single_chip_uses_default_meta():
    """Type A with mesh_chips=1 → use_shard_map=False (single chip, no sharding)."""
    from lib.expedition.expedition_worker_xla import _LoaderMeta

    instance = mock.MagicMock()  # has load_multichip_model
    instance.load_model = mock.MagicMock(return_value=mock.MagicMock())
    instance.load_multichip_model = mock.MagicMock(return_value=mock.MagicMock())

    item = _make_seed_item(mesh_chips=1)
    loader, meta = _call_build_loader_with_mock_instance(item, instance)

    assert meta.use_shard_map is False


def test_build_loader_xla_frontier_multichip_still_uses_default_meta():
    """Frontier model with mesh_chips=4 must NOT use shard_map — always _LoaderMeta()."""
    from lib.expedition.expedition_worker_xla import _build_loader_xla, _LoaderMeta, QueueItem

    item = QueueItem(
        model_id="org/gpt2-large",
        display_name="GPT2 Large",
        task="text-generation",
        source="huggingface",
        rarity="uncommon",
        hf_downloads=100000,
        hf_created_at=None,
        mesh_chips=4,  # multi-chip, but frontier always data-parallel
        loader_module=None,
        loader_class=None,
        is_frontier=True,
    )

    _, meta = _build_loader_xla(item)
    assert meta.use_shard_map is False, (
        "Frontier models always use data-parallel regardless of mesh_chips"
    )


def test_bench_passes_guard_in_run_worker_xla():
    """run_worker_xla must have 'item.mesh_chips == 1' guard for bench_passes."""
    import inspect
    import lib.expedition.expedition_worker_xla as worker
    source = inspect.getsource(worker.run_worker_xla)
    assert "item.mesh_chips == 1" in source, (
        "run_worker_xla must guard bench_passes with 'item.mesh_chips == 1'"
    )
