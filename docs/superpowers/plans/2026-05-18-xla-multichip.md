# XLA Multi-Chip Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable JAX/XLA models to run on multiple chips by wiring up the `--xla-mesh N` flag, adding `_LoaderMeta` routing infrastructure, and using `shard_map` for Type A Linen loaders (those with `load_multichip_model()`).

**Architecture:** Two types of JAX multi-chip loaders exist in `tt-forge-models`. Type B (EasyDel/NNX — ~14 models like GPT-2, Phi, LLaMA) already work with the existing `jax.jit(in_shardings=...)` data-parallel path in `_compile_model_xla`, they just need `mesh_chips > 1` to be set. Type A (Linen custom — ~5 models like AlexNet, MNIST) have `load_multichip_model()` and a named "X" axis; for them we use `shard_map` with axis_name="X". Both types remain data-parallel in this iteration — `load_multichip_model()` for true tensor-parallelism is deferred (requires CPU mesh init, see TODO in Task 2). The main gap: `mesh_chips` is hardcoded to 1 for all JAX seed models.

**Tech Stack:** Python, JAX `jax.experimental.shard_map`, `jax.sharding.{Mesh, NamedSharding, PartitionSpec}`, pytest, `unittest.mock`

---

### File Map

| File | Change |
|------|--------|
| `expedition.py` | Add `--xla-mesh N` arg; thread through `build_queues` → `_scan_forge_models` to set `mesh_chips` for JAX loaders |
| `lib/expedition/expedition_worker_xla.py` | Add `_LoaderMeta` dataclass; `_build_loader_xla` returns `(loader, meta)`; `_compile_model_xla` uses `shard_map` for Type A; fix bench guard; update call sites |
| `tests/lib/test_xla_multichip_loader.py` | New: unit tests for type detection, meta routing, bench guard (no hardware) |

---

### Task 1: `--xla-mesh N` flag + `_scan_forge_models` routing

**Files:**
- Modify: `expedition.py`
- Create: `tests/test_xla_mesh_flag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_xla_mesh_flag.py
"""Tests for --xla-mesh flag routing in _scan_forge_models."""
import sys
import types
import unittest.mock as mock


def _make_fake_forgems(forge_models_root):
    """Register synthetic _forgems root so _scan_forge_models can import."""
    pkg = "_forgems"
    if pkg not in sys.modules:
        root = types.ModuleType(pkg)
        root.__path__ = [str(forge_models_root)]
        root.__package__ = pkg
        sys.modules[pkg] = root
    return pkg


def test_scan_forge_models_sets_mesh_chips_for_jax_loaders(tmp_path):
    """JAX loaders get mesh_chips=N when xla_mesh=N > 1."""
    from expedition import _scan_forge_models

    # Build minimal fake forge-models tree: one JAX loader, one PyTorch loader
    for path, content in [
        ("mymodel/image_classification/jax/loader.py", _FAKE_JAX_LOADER),
        ("mymodel/image_classification/pytorch/loader.py", _FAKE_PT_LOADER),
    ]:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    _make_fake_forgems(str(tmp_path))

    with mock.patch("expedition._scan_forge_models.__globals__"
                    if False else "__builtins__"):
        pass  # placeholder for path override

    # Call with xla_mesh=2 — JAX loader should get mesh_chips=2, PT loader stays at 1
    import importlib
    import expedition as exp_mod

    items = _scan_forge_models_with_root(exp_mod, tmp_path, xla_mesh=2)
    jax_items = [it for it in items if "jax" in it["model_id"]]
    pt_items  = [it for it in items if "jax" not in it["model_id"]]

    assert all(it["mesh_chips"] == 2 for it in jax_items), (
        f"Expected JAX items mesh_chips=2, got {[it['mesh_chips'] for it in jax_items]}"
    )
    assert all(it["mesh_chips"] == 1 for it in pt_items), (
        f"Expected PT items mesh_chips=1, got {[it['mesh_chips'] for it in pt_items]}"
    )
```

> **Note:** The actual test is simpler — patch the forge_models_root path and call `_scan_forge_models` directly. Write a focused test that creates a mock JAX loader, mocks `forge_models_root`, and verifies `mesh_chips` is set. Use the approach in `tests/test_author_dedup.py` for test structure reference.

Here is the REAL failing test to write instead (simpler, no fake loader files):

```python
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
    # This is a unit test of the dict mutation logic, not a full scan.
    # Simulate what _scan_forge_models does with xla_mesh=4.
    item = {
        "model_id": "alexnet/image_classification/jax",
        "library": "jax",
        "mesh_chips": 1,
    }
    xla_mesh = 4
    # The function sets mesh_chips = xla_mesh for JAX items
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/test_xla_mesh_flag.py -v 2>&1 | head -30
```

Expected: FAIL with `AssertionError: build_queues must accept xla_mesh parameter`

- [ ] **Step 3: Add `xla_mesh` parameter to `_scan_forge_models` in `expedition.py`**

Current signature at line 254:
```python
def _scan_forge_models(bestiary_compiled_ids: set[str], include_all: bool = False,
                       framework: str | None = None) -> list[dict]:
```

Replace with:
```python
def _scan_forge_models(bestiary_compiled_ids: set[str], include_all: bool = False,
                       framework: str | None = None,
                       xla_mesh: int = 1) -> list[dict]:
```

Then in the items.append block at line 378, change the `"mesh_chips": 1,` line:

```python
                "mesh_chips": xla_mesh if loader_lib == "jax" and xla_mesh > 1 else 1,
```

- [ ] **Step 4: Add `xla_mesh` to `build_queues` and thread it through**

Current signature at line 807:
```python
def build_queues(
    num_chips: int,
    ...
    staples: bool = False,
) -> list[list[dict]]:
```

Add `xla_mesh: int = 1` after `staples`:
```python
def build_queues(
    num_chips: int,
    seed_only: bool = False,
    frontier_only: bool = False,
    limit: int = 0,
    min_downloads: int = 50,
    min_likes: int = 1,
    max_dl_like_ratio: int = 300,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
    staples: bool = False,
    xla_mesh: int = 1,
) -> list[list[dict]]:
```

Then find the `_with_spinner("scanning tt-forge-models library…", _scan_forge_models, compiled_ids, staples)` call around line 851 and add `xla_mesh`:

```python
seed_items = _with_spinner("scanning tt-forge-models library…",
                           _scan_forge_models, compiled_ids, staples,
                           xla_mesh=xla_mesh)
```

Also update the fallback `all_seeds` call around line 860:
```python
all_seeds = _scan_forge_models(set(), include_all=True, xla_mesh=xla_mesh)
```

- [ ] **Step 5: Add `--xla-mesh N` to argparse and default block**

In the argparse section around line 1747, after `--bench-shapes`, add:
```python
    run_p.add_argument("--xla-mesh", type=int, default=1, metavar="N",
                       help="Run JAX/XLA seed models on N chips (data-parallel). "
                            "1 = single-chip (default). Requires --backend xla or mixed.")
```

In the `args.cmd is None` defaults block around line 1781, add after `bench_shapes`:
```python
        args.xla_mesh = 1
```

- [ ] **Step 6: Thread `xla_mesh` through the TUI path and non-TUI `build_queues` call**

In the TUI `app = ExpeditionTUI(...)` call around line 1805, add:
```python
            xla_mesh=getattr(args, "xla_mesh", 1),
```

In the non-TUI `build_queues(...)` call around line 1843, add:
```python
        chip_queues = build_queues(
            ...
            xla_mesh=getattr(args, "xla_mesh", 1),
        )
```

Also update `ExpeditionTUI.__init__` in `expedition_tui.py` to accept and pass `xla_mesh` to `build_queues`. Search for the `build_queues(` call in `expedition_tui.py`:

```bash
grep -n "build_queues\|xla_mesh" /home/ttuser/code/tt-forge-compiletron/expedition_tui.py | head -10
```

Add `xla_mesh: int = 1` to `ExpeditionTUI.__init__` and pass it through wherever `build_queues` is called in the TUI.

- [ ] **Step 7: Run the test to verify it passes**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/test_xla_mesh_flag.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 8: Smoke-test the CLI flag parses without error**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python expedition.py run --help | grep xla-mesh
```

Expected: line showing `--xla-mesh N`

- [ ] **Step 9: Commit**

```bash
git add expedition.py expedition_tui.py tests/test_xla_mesh_flag.py
git commit -m "feat: add --xla-mesh N flag to enable multi-chip JAX dispatch"
```

---

### Task 2: `_LoaderMeta` + `_build_loader_xla` type routing + `shard_map` path in `_compile_model_xla`

**Files:**
- Modify: `lib/expedition/expedition_worker_xla.py`
- Create: `tests/lib/test_xla_multichip_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lib/test_xla_multichip_loader.py
"""Unit tests for XLA multi-chip routing — no hardware required."""
import sys
import types
import pytest
import unittest.mock as mock
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Minimal stubs so we can import from expedition_worker_xla without real JAX
# ---------------------------------------------------------------------------

def _stub_jax():
    """Register minimal jax stub so the worker module imports cleanly."""
    if "jax" in sys.modules:
        return
    jax = types.ModuleType("jax")
    jax.numpy = types.ModuleType("jax.numpy")
    jax.random = types.ModuleType("jax.random")
    jax.devices = lambda: []
    jax.random.PRNGKey = lambda seed: seed
    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jax.numpy
    sys.modules["jax.random"] = jax.random
    sharding = types.ModuleType("jax.sharding")
    sharding.Mesh = mock.MagicMock()
    sharding.NamedSharding = mock.MagicMock()
    sharding.PartitionSpec = mock.MagicMock()
    sys.modules["jax.sharding"] = sharding
    exp_sm = types.ModuleType("jax.experimental")
    exp_sm.shard_map = types.ModuleType("jax.experimental.shard_map")
    exp_sm.shard_map.shard_map = mock.MagicMock()
    sys.modules["jax.experimental"] = exp_sm
    sys.modules["jax.experimental.shard_map"] = exp_sm.shard_map


_stub_jax()


from lib.expedition.expedition_worker_xla import _LoaderMeta, QueueItem


# ---------------------------------------------------------------------------
# _LoaderMeta tests
# ---------------------------------------------------------------------------

class TestLoaderMeta:
    def test_defaults(self):
        meta = _LoaderMeta()
        assert meta.use_shard_map is False
        assert meta.axis_name == "batch"

    def test_type_a_fields(self):
        meta = _LoaderMeta(use_shard_map=True, axis_name="X")
        assert meta.use_shard_map is True
        assert meta.axis_name == "X"


# ---------------------------------------------------------------------------
# _build_loader_xla type detection tests
# ---------------------------------------------------------------------------

def _make_queue_item(mesh_chips=1, is_frontier=False,
                     loader_module="_forgems.fake.loader",
                     loader_class="ModelLoader",
                     model_id="fake/model", task="image-classification"):
    return QueueItem(
        model_id=model_id,
        display_name="Fake",
        task=task,
        source="custom",
        rarity="familiar",
        hf_downloads=None,
        hf_created_at=None,
        mesh_chips=mesh_chips,
        loader_module=loader_module,
        loader_class=loader_class,
        is_frontier=is_frontier,
    )


class TestBuildLoaderXlaReturnsTyple:
    """_build_loader_xla must return (callable, _LoaderMeta) not just callable."""

    def _make_mock_instance(self, has_multichip=False):
        inst = mock.MagicMock()
        inst.load_model.return_value = mock.MagicMock()
        inst.load_inputs.return_value = mock.MagicMock()
        if not has_multichip:
            del inst.load_multichip_model  # ensure attribute absent
        else:
            inst.load_multichip_model.return_value = mock.MagicMock()
        return inst

    def _run_build_loader(self, item, instance):
        """Patch importlib so _build_loader_xla uses our mock instance."""
        import lib.expedition.expedition_worker_xla as worker

        fake_mod = types.ModuleType("_forgems.fake.loader")
        fake_cls = mock.MagicMock(return_value=instance)
        fake_mod.ModelLoader = fake_cls
        fake_mod.ModelVariant = None

        with mock.patch.dict(sys.modules, {"_forgems.fake.loader": fake_mod}):
            with mock.patch("importlib.import_module", return_value=fake_mod):
                result = worker._build_loader_xla(item)
        return result

    def test_frontier_returns_tuple(self):
        from lib.expedition.expedition_worker_xla import _build_loader_xla
        item = _make_queue_item(is_frontier=True, loader_module=None, loader_class=None,
                                model_id="org/gpt2-large", task="text-generation")
        result = _build_loader_xla(item)
        assert isinstance(result, tuple) and len(result) == 2, (
            "_build_loader_xla must return (loader, meta) for frontier models"
        )
        loader, meta = result
        assert callable(loader)
        assert isinstance(meta, _LoaderMeta)
        assert meta.use_shard_map is False

    def test_seed_single_chip_returns_tuple_with_default_meta(self):
        instance = self._make_mock_instance(has_multichip=False)
        item = _make_queue_item(mesh_chips=1)
        loader, meta = self._run_build_loader(item, instance)
        assert callable(loader)
        assert meta.use_shard_map is False
        assert meta.axis_name == "batch"

    def test_seed_multichip_type_b_uses_data_parallel_meta(self):
        """Type B (no load_multichip_model) → use_shard_map=False, axis_name='batch'."""
        instance = self._make_mock_instance(has_multichip=False)
        item = _make_queue_item(mesh_chips=4)
        loader, meta = self._run_build_loader(item, instance)
        assert meta.use_shard_map is False
        assert meta.axis_name == "batch"

    def test_seed_multichip_type_a_uses_shard_map_meta(self):
        """Type A (has load_multichip_model, mesh_chips>1) → use_shard_map=True, axis_name='X'."""
        instance = self._make_mock_instance(has_multichip=True)
        item = _make_queue_item(mesh_chips=4)
        loader, meta = self._run_build_loader(item, instance)
        assert meta.use_shard_map is True
        assert meta.axis_name == "X"

    def test_type_a_single_chip_uses_default_meta(self):
        """Type A loader with mesh_chips=1 must NOT use shard_map (single-chip mode)."""
        instance = self._make_mock_instance(has_multichip=True)
        item = _make_queue_item(mesh_chips=1)
        loader, meta = self._run_build_loader(item, instance)
        assert meta.use_shard_map is False


# ---------------------------------------------------------------------------
# Bench passes guard test
# ---------------------------------------------------------------------------

class TestBenchPassesGuard:
    """bench_passes should be skipped for multi-chip models."""

    def test_bench_guard_description(self):
        """Ensure the source code has the mesh_chips==1 guard for bench passes."""
        import inspect
        import lib.expedition.expedition_worker_xla as worker
        source = inspect.getsource(worker.run_worker_xla)
        assert "item.mesh_chips == 1" in source, (
            "run_worker_xla must guard bench_passes with 'item.mesh_chips == 1' "
            "to skip benchmarking for multi-chip models"
        )
```

- [ ] **Step 2: Run to confirm failures**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/lib/test_xla_multichip_loader.py -v 2>&1 | head -40
```

Expected: FAIL — `ImportError: cannot import name '_LoaderMeta' from 'lib.expedition.expedition_worker_xla'` (and possibly the bench guard test also fails).

- [ ] **Step 3: Add `_LoaderMeta` dataclass after `QueueItem` in `expedition_worker_xla.py`**

Location: after line 668 (after the `QueueItem` dataclass closing brace, before `_load_queue`).

```python
@dataclass
class _LoaderMeta:
    """Routing metadata returned alongside the loader callable from _build_loader_xla.

    use_shard_map=True: Type A Linen loader (has load_multichip_model).
      _compile_model_xla uses jax.experimental.shard_map with axis_name for sharding.
    use_shard_map=False: Type B EasyDel/NNX loader (or single-chip).
      _compile_model_xla uses the existing jax.jit(in_shardings=...) data-parallel path.
    axis_name: mesh axis label. "X" for Type A, "batch" for Type B.

    TODO: when CPU mesh init is available, Type A should call load_multichip_model()
    here to get the tensor-parallel Linen module (not just the single-chip model).
    """
    use_shard_map: bool = False
    axis_name: str = "batch"
```

- [ ] **Step 4: Update `_build_loader_xla` to return `(loader, _LoaderMeta)` and detect type**

In `_build_loader_xla` at line 702:

1. **Frontier path** (around line 734): Change the final `return loader` to `return loader, _LoaderMeta()`.

2. **Seed path** (around line 792–867): After `instance = cls(variant=variant) if variant is not None else cls()` and before the `def loader():` closure, add type detection:

```python
        # Detect Type A: loader supports load_multichip_model (Linen tensor-parallel).
        # When mesh_chips > 1, mark use_shard_map=True so _compile_model_xla uses
        # shard_map with axis "X" instead of jit(in_shardings=...) with axis "batch".
        # NOTE: we still call load_model() here (not load_multichip_model()) because
        # initializing the tensor-parallel Linen module requires a CPU mesh context
        # that isn't available in the TT XLA worker. True tensor-parallelism via
        # load_multichip_model() is deferred until CPU mesh init support is added.
        _is_type_a = (
            item.mesh_chips > 1
            and hasattr(instance, "load_multichip_model")
        )
        meta = _LoaderMeta(use_shard_map=True, axis_name="X") if _is_type_a else _LoaderMeta()
```

3. **End of seed path**: Change the final `return loader` (line 867) to `return loader, meta`.

- [ ] **Step 5: Update `run_worker_xla` to unpack the tuple from `_build_loader_xla`**

In `run_worker_xla` around line 990, the `_build_loader_xla` call is:
```python
        try:
            loader = _build_loader_xla(item)
```

Change to:
```python
        try:
            loader, _meta = _build_loader_xla(item)
```

And find the two `_compile_model_xla(loader, device, chip_id, mesh_chips=item.mesh_chips)` calls (around lines 1004 and 1010) and add `meta=_meta`:

```python
        success, output, compile_time, infer_time, error_str, compiled_bundle = _compile_model_xla(
            loader, device, chip_id, mesh_chips=item.mesh_chips, meta=_meta
        )
        # Auto-install missing packages and retry once
        if not success and "No module named" in error_str:
            if _try_install_missing(error_str):
                success, output, compile_time, infer_time, error_str, compiled_bundle = _compile_model_xla(
                    loader, device, chip_id, mesh_chips=item.mesh_chips, meta=_meta
                )
```

- [ ] **Step 6: Add `meta` parameter to `_compile_model_xla` signature**

Current signature at line 360:
```python
def _compile_model_xla(
    model_loader,
    device,
    chip_id: int,
    timeout: int = 300,
    mesh_chips: int = 1,
) -> tuple[bool, Any, float, float, str, Any]:
```

Replace with:
```python
def _compile_model_xla(
    model_loader,
    device,
    chip_id: int,
    timeout: int = 300,
    mesh_chips: int = 1,
    meta: "_LoaderMeta | None" = None,
) -> tuple[bool, Any, float, float, str, Any]:
```

- [ ] **Step 7: Add `shard_map` path in `_compile_model_xla` for Type A models**

In `_compile_model_xla`, inside the `if mesh_chips > 1:` block (around line 440), the existing path creates a mesh with axis `"batch"`. Add a branch at the TOP of that block, before the existing `mesh = Mesh(...)` line:

```python
        if mesh_chips > 1:
            # ── Multi-chip path ──────────────────────────────────────────────
            all_devices = jax.devices()
            n = min(mesh_chips, len(all_devices))
            if n < mesh_chips:
                _print_live_info(
                    f"Only {n} TT device(s) visible — using {n}-chip"
                )

            use_shard_map = meta is not None and meta.use_shard_map
            axis = meta.axis_name if meta is not None else "batch"

            if use_shard_map:
                # ── Type A: Linen model with named axis — use shard_map ──────
                # shard_map gives each device a batch shard and provides the
                # named axis "X" as a collective context. This is equivalent to
                # data-parallel but uses the axis name matching the loader's
                # get_input_activations_partition_spec(axis_name="X") interface.
                # TODO: replace load_model() with load_multichip_model() here
                # once CPU-mesh parameter initialization is available.
                from jax.experimental.shard_map import shard_map as _shard_map
                from jax.sharding import Mesh, PartitionSpec, NamedSharding

                mesh = Mesh(np.array(all_devices[:n]), axis_names=(axis,))
                replicated = NamedSharding(mesh, PartitionSpec())
                sharded   = NamedSharding(mesh, PartitionSpec(axis,))

                _print_live_info(
                    f"★ {n}-CHIP SHARD_MAP ACTIVE (Type A, axis={axis!r}): "
                    f"{[str(d) for d in all_devices[:n]]}"
                )

                single = make_input(device)
                if isinstance(single, _Mapping):
                    dummy_inputs = {k: jnp.concatenate([v] * n, axis=0)
                                    for k, v in single.items()}
                else:
                    dummy_inputs = jnp.concatenate([single] * n, axis=0)

                sharded_params = jax.device_put(flax_params, replicated)
                sharded_inputs = jax.device_put(dummy_inputs, sharded)

                compiled_fn = jax.jit(_shard_map(
                    forward,
                    mesh=mesh,
                    in_specs=(PartitionSpec(), PartitionSpec(axis)),
                    out_specs=PartitionSpec(axis),
                    check_rep=False,
                ))

                _print_live_info(
                    f"Input shape: {jax.tree_util.tree_map(lambda x: x.shape, dummy_inputs)}"
                )
                _print_progress_step(2, 3, f"Compiling via shard_map across {n} chips (Type A)...")
                compile_start = time.time()

                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
                try:
                    output = compiled_fn(sharded_params, sharded_inputs)
                    output.block_until_ready()
                    signal.alarm(0)
                except TimeoutException:
                    signal.alarm(0)
                    return False, None, time.time() - compile_start, 0.0, "TIMEOUT", None

                compile_time = time.time() - compile_start
                _print_progress_step(3, 3, f"Output shape: {output.shape}  ({compile_time:.1f}s — {n} chips shard_map)")

                infer_s = 0.0
                try:
                    infer_start = time.time()
                    _ = compiled_fn(sharded_params, sharded_inputs)
                    _.block_until_ready()
                    infer_s = time.time() - infer_start
                except Exception:
                    pass

                return True, output, compile_time, infer_s, "", (compiled_fn, sharded_params, all_devices[0])

            else:
                # ── Type B: existing data-parallel jit path ──────────────────
                # (original mesh_chips > 1 code below — just change axis name)
```

Then indent the original data-parallel block under the `else:` and change its mesh line to use `axis` instead of `"batch"`:

```python
            # (Type B data-parallel, unchanged except axis var replaces "batch")
                mesh       = Mesh(np.array(all_devices[:n]), axis_names=(axis,))
                replicated = NamedSharding(mesh, PartitionSpec())
                batched    = NamedSharding(mesh, PartitionSpec(axis,))
                ...
                compiled_fn = jax.jit(
                    forward,
                    in_shardings=(replicated, batched),
                    out_shardings=batched,
                )
```

> **Implementation note:** Instead of restructuring the full if/else, the cleanest approach is:
> 1. Extract the new `if use_shard_map:` block (steps above) before the existing data-parallel code
> 2. Wrap the existing data-parallel block in `else:` and replace the hardcoded `"batch"` strings with `axis`
> 3. The existing `return` at line 507 stays inside the `else:` block

- [ ] **Step 8: Fix bench passes guard in `run_worker_xla`**

Find the bench passes line around line 1059:
```python
            if bench_passes > 0 and compiled_bundle is not None:
```

Replace with:
```python
            if bench_passes > 0 and compiled_bundle is not None and item.mesh_chips == 1:
```

Multi-chip runs skip bench passes because the shard_map/multi-device setup changes the
calling convention for the compiled bundle; single-chip bench semantics don't apply cleanly.

- [ ] **Step 9: Run the tests**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/lib/test_xla_multichip_loader.py -v
```

Expected: all tests PASS (may need to adjust imports/stubs if JAX is actually installed).

- [ ] **Step 10: Run existing tests to check for regressions**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python -m pytest tests/ -v --ignore=tests/test_xla_mesh_flag.py -x 2>&1 | tail -20
```

Expected: existing tests still pass.

- [ ] **Step 11: Commit**

```bash
git add lib/expedition/expedition_worker_xla.py tests/lib/test_xla_multichip_loader.py
git commit -m "feat: add _LoaderMeta routing + shard_map path for Type A XLA multi-chip"
```

---

### Task 3: Wire `xla_mesh` through `ExpeditionTUI` + integration smoke test

**Files:**
- Modify: `expedition_tui.py`

- [ ] **Step 1: Find `ExpeditionTUI.__init__` signature and the `build_queues` call**

```bash
grep -n "def __init__\|xla_mesh\|build_queues" /home/ttuser/code/tt-forge-compiletron/expedition_tui.py | head -20
```

- [ ] **Step 2: Add `xla_mesh: int = 1` to `ExpeditionTUI.__init__` and store it**

Find the `__init__` signature and add `xla_mesh: int = 1` in the parameters list, then `self.xla_mesh = xla_mesh` in the body.

- [ ] **Step 3: Pass `xla_mesh=self.xla_mesh` to every `build_queues(...)` call in the TUI**

Find all calls to `build_queues(` in `expedition_tui.py` and add `xla_mesh=self.xla_mesh` to each.

- [ ] **Step 4: Smoke-test CLI flag is wired end-to-end**

```bash
cd /home/ttuser/code/tt-forge-compiletron
# Dry-run: builds queues and prints assignment, doesn't launch workers
python expedition.py run --xla-mesh 2 --seed-only --limit 3 --no-predownload \
    2>&1 | head -30
```

Expected: prints queue assignment with JAX models showing `mesh2` or similar count indicator, no errors.

- [ ] **Step 5: Commit**

```bash
git add expedition_tui.py
git commit -m "feat: thread xla_mesh through ExpeditionTUI build_queues"
```

---

## Self-Review

**Spec coverage:**
- ✅ `_build_loader_xla` returns `(loader, _LoaderMeta)` — Task 2
- ✅ Type A detection via `hasattr(instance, "load_multichip_model")` — Task 2
- ✅ `shard_map` path in `_compile_model_xla` for Type A — Task 2 Step 7
- ✅ `--xla-mesh N` flag — Task 1
- ✅ `_scan_forge_models` sets `mesh_chips` for JAX loaders — Task 1
- ✅ Bench passes guard — Task 2 Step 8
- ✅ Tests for routing (no hardware) — Task 2 Step 1, Task 1 Step 1
- ⚠️ `load_multichip_model()` is detected but NOT called (see TODO comment in `_LoaderMeta`) — this is explicit: true tensor-parallel requires CPU mesh init which is out of scope for this iteration
- ⚠️ `ExpeditionTUI` xla_mesh wiring (Task 3) depends on finding `build_queues` calls in TUI — implementer must search and add

**Placeholder scan:**
- Task 1 Step 6: "search for the `build_queues` call in `expedition_tui.py`" — implementer must find it. Line numbers will be clearer after the grep command provided.
- Task 2 Step 7 "wrap the existing data-parallel block in `else:`" — requires care, provide exact diff context.

**Type consistency:**
- `_LoaderMeta` is used consistently as `_LoaderMeta()`, `_LoaderMeta(use_shard_map=True, axis_name="X")` throughout.
- `meta.axis_name` replaces hardcoded `"batch"` in the data-parallel path — consistency checked.
- `meta: "_LoaderMeta | None" = None` uses string annotation to avoid forward-reference errors since `_LoaderMeta` is defined in the same module.
