# Compilation Pipelines — Technical Reference

How tt-forge-compiletron loads, compiles, and runs models on Tenstorrent Blackhole hardware.
Two independent backends share a queue and bestiary but diverge completely in implementation.

---

## Architecture overview

```
expedition.py
    └── builds queue (QueueItem list)
            ├── forge worker  — subprocess per chip → expedition_worker.py
            └── xla worker    — subprocess per chip → expedition_worker_xla.py
```

Each worker subprocess owns one TT chip and iterates through the queue independently.
Results are written to per-chip CSV files and merged into `data/bestiary.json`.

---

## Backend 1 — forge (PyTorch)

Entry point: `lib/expedition/expedition_worker.py :: run_worker()`

### Loader construction — `_build_loader(item)`

Two code paths depending on `item.is_frontier`:

**Frontier models** (live HuggingFace discovery):
`hf_discover.build_dynamic_loader()` auto-detects architecture from the model's
`config.json` and returns a closure that calls `AutoModelForXxx.from_pretrained()`.

**Seed models** (from `~/code/tt-forge-models`):
1. Registers a synthetic `_forgems` root package pointing at the models repo
   (the directory name has a hyphen so it can't be a real Python package).
2. Imports the loader class via `importlib.import_module(item.loader_module)`.
3. Wraps `instance.load_model()` in a closure.
4. Attaches `_input_type` ("text" / "audio" / "image") derived from the task string.
5. Attaches `_load_inputs` if the loader provides `load_inputs()` for structured inputs.

### Compile pipeline — `_compile_model(model_loader, chip_id)`

```
model_loader()          → torch.nn.Module
model.eval()
config patch:           use_cache=False, return_dict=False,
                        output_attentions=False, output_hidden_states=False
sample input selection:
    1. loader.load_inputs()         (structured, matches forward signature)
    2. encoder-decoder guard        (input_ids + decoder_input_ids for T5/MusicGen/BART)
    3. task-name heuristic          (text=randint(0,1000,(1,32)), audio=randn(1,16000),
                                     image=randn(1,3,224,224))
_normalise_inputs()     → contiguous float32 tensors (PIL→numpy pipeline can
                          produce strided non-contiguous tensors)
forge.compile(model, sample_inputs)
    └── retry with _LogitsWrapper   if "Tracer cannot infer type" (tuple/ModelOutput return)
compiled(*infer_inputs) → output tensor
```

**Config patches explained:**
- `use_cache=False` — KV caches have dynamic shapes; disabling forces a single logits tensor.
- `return_dict=False` — forge's TorchScript tracer requires tuples, not HuggingFace ModelOutput dataclasses.
- `output_attentions/hidden_states=False` — extra output tensors confuse the forge tracer on retry.

**_LogitsWrapper retry:**
HuggingFace causal-LM models commonly return `CausalLMOutputWithPast` or `(logits, past_kv)`.
The forge tracer fails with "Tracer cannot infer type" on mixed-type tuples. On first failure,
the model is wrapped in `_LogitsWrapper` which strips the return to `out[0]` or `out.logits`.
This converts a large class of frontier models from guaranteed failure to likely success.

**Encoder-decoder guard:**
Models with `config.is_encoder_decoder=True` (MusicGen, T5, BART) need `decoder_input_ids`.
Without it, `ones_like()` receives `None` → crash. The guard provides both tensors when
`load_inputs()` is absent and no encoder-decoder-aware inputs are available.

### First Voice — `_attempt_first_voice(compiled, sample_inputs, tokenizer)`

After a successful compile, runs the compiled model with real tokenized text and decodes
the top-3 token predictions. Stored in the bestiary as `first_voice` — proof the model
actually runs, not just compiles.

---

## Backend 2 — XLA (JAX/Flax)

Entry point: `lib/expedition/expedition_worker_xla.py :: run_worker_xla()`
Runtime: the `xla-venv` virtualenv with `pjrt-plugin-tt 0.9.0` + `jax==0.7.1`.

### JAX/TT device setup — `_setup_jax(chip_id)`

```python
os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)
os.environ["TT_MESH_GRAPH_DESC_PATH"] = ".../p150_mesh_graph_descriptor.textproto"
import jax  # triggers PJRT plugin registration
device = jax.devices()[0]
```

Two compat patches applied at import time (`_apply_jax_compat_patches`):
- `jax.core.cur_sublevel` shim — PJRT plugin assumes an API that changed in JAX 0.4.x.
- `jax.local_devices()` shim — returns only `[device]` for the chip this worker owns.

### Loader construction — `_build_loader_xla(item)`

**Frontier models:**
Maps `item.task` to one of `FlaxAutoModelForCausalLM / SeqToSeqLM / MaskedLM / ImageClassification`.

```python
result = FlaxAutoModelForCausalLM.from_pretrained(model_id, dtype="float32", _do_init=False)
model, params = result  # _do_init=False always returns a (model, params) tuple
model.params = params   # reattach for internal forward accesses
```

`_do_init=False` is essential: HuggingFace's Flax eager init runs SliceOps on the device,
which pjrt-plugin-tt 0.9.0 does not support in eager mode. JIT-compiled SliceOps work fine.

**Seed models (from tt-forge-models JAX loaders):**

1. Same `_forgems` synthetic package trick as the forge worker.
2. Patches `FlaxPreTrainedModel.from_pretrained` to inject `_do_init=False` by default —
   the seed loaders call `from_pretrained` internally and need the same protection.

   ```python
   @classmethod
   def _patched(cls, *args, **kw):
       if kw.get("from_pt", False):
           return _orig_func(cls, *args, **kw)  # from_pt handles init itself
       kw.setdefault("_do_init", False)
       try:
           return _orig_func(cls, *args, **kw)
       except ValueError as ve:
           if "params" in str(ve) and "_do_init" in str(ve):
               # No native Flax checkpoint — retry with PyTorch→Flax conversion.
               kw.pop("_do_init", None)
               kw["from_pt"] = True
               return _orig_func(cls, *args, **kw)
           raise
   ```

   **Critical restore detail:** `_orig = _FPTM.from_pretrained` captures a bound classmethod.
   Restoring `_FPTM.from_pretrained = _orig` writes a bound method as a plain attribute.
   Subclasses resolving `from_pretrained` via MRO then get `cls=FlaxPreTrainedModel` baked in,
   breaking their own `__init__` signatures. Must restore as a proper descriptor:
   ```python
   _FPTM.from_pretrained = classmethod(_orig_func)  # not = _orig
   ```

3. EasyDel loaders (17 of the 18 JAX seed loaders) use `AutoEasyDeLModelForCausalLM`
   which returns a `flax.nnx.Module`. These don't populate `config._name_or_path`, so
   the loader stashes the pretrained name explicitly:
   ```python
   object.__setattr__(model, "_compiletron_pretrained_name", instance._model_name)
   ```

4. Returns `(loader_fn, _LoaderMeta)`. `_LoaderMeta.use_shard_map=True` flags Linen models
   with `load_multichip_model()` for the shard_map multi-chip path.

### Compile pipeline — `_compile_model_xla(model_loader, device, chip_id, mesh_chips)`

```
model_loader()   → (model, params, tokenizer, make_input_fn)
```

`make_input_fn(device) → dict[str, jax.Array]` — produces the dummy input for JIT tracing.

#### Forward function selection (four calling conventions)

Flax/JAX models have three distinct calling conventions. The worker detects which applies
and builds a matching `forward(params, inputs)` closure:

**Type A — HuggingFace `FlaxPreTrainedModel`** (e.g. GPT-2 Flax, BLOOM, BERT):
```python
def forward(params, inputs):
    out = model(**inputs, params=params, train=False)
    return out.logits or out[0] or out
```
Weights are kept separate from the model object and passed explicitly at call time.
This is by design: `_do_init=False` skips eager init and returns params as a separate pytree.

**Type B — Flax Linen `.apply()`** (e.g. AlexNet, DINOv2, custom Linen modules):
```python
def forward(params, inputs):
    var_coll = params if "params" in params else {"params": params}
    out = model.apply(var_coll, **inputs)  # or positional for raw-array inputs
    return out.logits or out[0] or out
```
Linen modules pass the variable collection (params + any batch_stats) through `.apply()`.
The `var_coll` check handles both raw param trees and full variable collections.

**Type C — EasyDel NNX `flax.nnx.Module`** (17 EasyDel-based JAX loaders):

EasyDel bakes a 5D internal mesh (`dp/fsdp/tp/sp/expert`) into every model.
TT MLIR only supports 1D/2D mesh — any `with_sharding_constraint` referencing the 5D mesh
crashes the StableHLO lowering pipeline with `error: Pass expects a 1D or 2D mesh, got 5D`.

**Primary strategy — HF Flax fallback:**
The EasyDel model has already downloaded the checkpoint to HuggingFace cache.
Reload the equivalent native `FlaxAutoModelForCausalLM` from that same cache:
```python
_hf_result = FlaxAutoModelForCausalLM.from_pretrained(_pretrained_name, _do_init=False)
model, params = _hf_result  # now a clean FlaxPreTrainedModel, no EasyDel mesh
```
This gives a Type A model with no sharding infrastructure. The pretrained name is read from
`model._compiletron_pretrained_name` (stashed by the loader — EasyDel leaves `config._name_or_path` empty).

**Last resort — `nnx.split/merge` with sharding patch:**
If the HF fallback fails, split the EasyDel model into a static graphdef + parameter pytree,
then patch `eformer.escale.with_sharding_constraint` (EasyDel's sharding entry point) to a
no-op during the forward pass:
```python
_graphdef, _nnx_state = nnx.split(model)
# during forward:
_escale.with_sharding_constraint = lambda x, *_a, **_kw: x
_m = nnx.merge(_graphdef, params)
out = _m(**inputs)
```

#### JIT compilation

```python
compiled_fn = jax.jit(forward)
output = compiled_fn(flax_params, dummy_inputs)
output.block_until_ready()   # first call = XLA compile + inference
# compile_time = wall clock of this call (includes both)

_ = compiled_fn(flax_params, dummy_inputs)
_.block_until_ready()        # second call = pure inference (JIT already warm)
# infer_s = wall clock of this call
```

Unlike forge, there is no explicit "compile" step. `jax.jit` traces lazily; the first call
compiles the XLA program and executes it. The reported `compile_time` covers both.

#### Multi-chip paths (mesh_chips > 1)

**Type A (Linen, `use_shard_map=True`) — `shard_map`:**
```python
mesh = Mesh(np.array(all_devices[:n]), axis_names=("X",))
compiled_fn = jax.jit(shard_map(forward, mesh=mesh,
    in_specs=(PartitionSpec(), PartitionSpec("X")),
    out_specs=PartitionSpec("X"),
    check_rep=False))
# params replicated, batch sharded: each chip gets 1/n of the batch
```

**Type B (HF Flax, `use_shard_map=False`) — `jit(in_shardings=...)`:**
```python
mesh = Mesh(np.array(all_devices[:n]), axis_names=("batch",))
replicated = NamedSharding(mesh, PartitionSpec())
batched    = NamedSharding(mesh, PartitionSpec("batch"))
compiled_fn = jax.jit(forward,
    in_shardings=(replicated, batched),
    out_shardings=batched)
```

Both paths replicate params and shard the input batch. XLA handles cross-device
communication automatically via the PJRT collective ops.

---

## Model loader interface (tt-forge-models)

The `~/code/tt-forge-models` repo provides loader classes. The compilation workers
discover them by `loader_module` + `loader_class` from the queue item.

### Forge (PyTorch) loader contract

```python
class ModelLoader:
    def load_model(self) -> torch.nn.Module:
        ...
    def load_inputs(self) -> dict | list | torch.Tensor:  # optional
        ...
    _input_type: str  # optional: "text" | "audio" | "image"
```

### XLA (JAX/Flax) loader contract

```python
class ModelLoader:
    _model_name: str                  # HF pretrained name, e.g. "openai-community/gpt2"
    def load_model(self) -> model | (model, params):
        ...
    def load_inputs(self) -> dict[str, jax.Array]:  # optional
        ...
    def load_parameters(self) -> dict:              # optional; for Linen custom-init models
        ...
    def _load_tokenizer(self) -> transformers.PreTrainedTokenizer:  # optional
        ...
    def load_multichip_model(self) -> ...:          # optional; flags Type A shard_map path
        ...
```

`load_model()` may return either a bare model or a `(model, params)` tuple.
The worker handles both shapes — params are extracted from `model.params` if not in the tuple.

### EasyDel loader pattern (18 JAX seed loaders)

All EasyDel-based loaders follow the same structure:

```python
from easydel import AutoEasyDeLModelForCausalLM, PartitionSpec

class ModelLoader:
    _model_name = "openai-community/gpt2"

    def load_model(self):
        model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name,
            partition_rules=((r".*", PartitionSpec()),),
        )
        return model, params
```

The XLA worker intercepts these and reloads via `FlaxAutoModelForCausalLM` to bypass
EasyDel's 5D mesh. The `_model_name` attribute is the critical link — it's stashed on the
returned model object so the worker can find the checkpoint in HF cache.

---

## Using this code as middleware for Forge/XLA

The key patterns for loading JAX/Flax models on TT hardware:

1. **`_do_init=False`** — always pass this to `FlaxXxx.from_pretrained()`. Without it,
   HuggingFace's eager Flax init fails on TT hardware (SliceOp not supported in eager mode).
   JIT-compiled SliceOps work fine.

2. **Tuple return handling** — `_do_init=False` makes `from_pretrained` return `(model, params)`.
   Reattach params for models that read `self.params` internally:
   ```python
   model, params = FlaxAutoModelForCausalLM.from_pretrained(name, _do_init=False)
   try:
       model.params = params
   except Exception:
       object.__setattr__(model, '_params', params)
   ```

3. **EasyDel → HF Flax swap** — if you need to compile an EasyDel model, reload the
   equivalent HF Flax model from the same cache instead. EasyDel's 5D mesh is incompatible
   with TT MLIR's 1D/2D mesh constraint.

4. **Forward function shape** — wrap your model call as `forward(params, inputs)` for `jax.jit`.
   Keep params and inputs as separate args; JAX traces them as distinct pytrees.

5. **`block_until_ready()`** — always call this on the output to ensure the TT device
   has actually finished execution before measuring time or reading results.
