# KV Cache Benchmarking — How It Works

## Why Two Compiled Graphs?

For causal language models (GPT-2, Falcon, LLaMA, etc.) there are two fundamentally
different phases of inference:

| Phase | Inputs | Output |
|-------|--------|--------|
| **Prefill** | Full prompt (N tokens at once) | First token's logits |
| **Decode** | Single new token + K/V cache | Next token's logits |

Running decode as another prefill (processing the full growing context) wastes
hardware — each new token forces a full forward pass over all previous tokens.

With KV cache, decode processes **one token at a time** and reads the cached K/V
state from previous positions.  On real autoregressive generation, this is 10–50×
faster than full recompute.

## Why StaticCache?

`transformers.DynamicCache` grows each step (the K/V tensors get longer).  forge's
compile model requires **fixed tensor shapes** — it traces a single graph at compile
time, and all subsequent calls must have the same shapes.

`transformers.StaticCache` pre-allocates a fixed `[batch, heads, max_cache_len, head_dim]`
buffer per layer.  At each decode step, one slot is overwritten (via `cache.update()`).
The shapes never change — the same compiled decode graph runs for every token.

## The Two-Graph Pattern in bench_decode.py

```
     model (Python)
          │
          ├─────────────────────────────────────────────────────┐
          │                                                     │
     Prefill wrapper                                    KVDecodeWrapper
     (use_cache=False)                                  (embeds StaticCache)
          │                                                     │
   forge.compile() →                               forge.compile() →
   compiled_pf                                     compiled_dec
          │                                                     │
   TTFT / prefill tok/s                           decode tok/s (1 tok/step)
```

### Prefill Graph

```python
class Single(torch.nn.Module):
    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False, return_dict=False)
        return out[0]

compiled_pf = forge.compile(Single(model), sample_inputs=[prompt_ids])
```

### Decode Graph

```python
class KVDecodeWrapper(torch.nn.Module):
    def __init__(self, model, static_cache):
        super().__init__()
        self.model = model
        self.kv_cache = static_cache   # StaticCache is a torch.nn.Module submodule

    def forward(self, input_ids, cache_position):
        out = self.model(
            input_ids=input_ids,
            past_key_values=self.kv_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=False,
        )
        return out[0]

# Pre-fill the cache on CPU (uses tools/utils.py helper)
dec_inputs = get_static_cache_decode_inputs(
    tokenizer=tok, config=model.config, model=model,
    batch_size=1, max_cache_len=decode_len, device="cpu",
)
kv_wrapper = KVDecodeWrapper(model, dec_inputs["past_key_values"])
compiled_dec = forge.compile(kv_wrapper, sample_inputs=[
    dec_inputs["input_ids"],        # (1, 1) long tensor
    dec_inputs["cache_position"],   # (1,)   long tensor
])
```

## Why the Cache is a Submodule, Not a Forward Arg

forge's `sample_inputs` must all be `torch.Tensor` objects.  `StaticCache` is a
`torch.nn.Module` (not a tensor), so it can't be passed as a sample_input.

By embedding it as `self.kv_cache`, forge registers the K/V tensors as **model
state** (like parameters and buffers) rather than runtime inputs.  When forge
traces the forward pass and sees `self.kv_cache.update(...)`, it emits
`FillCache` / `UpdateCache` ops in the compiled IR instead:

```
FillCache(name, cache_tensor, input_tensor, batch_offset)
UpdateCache(name, cache_tensor, input_tensor, update_index, batch_offset)
```

These ops write to the fixed-shape cache buffers in-place without dynamic shapes.

## Benchmark Methodology

For timing we keep `cache_position` **fixed** at `max_cache_len - 1`.  Each decode
step overwrites the same cache slot, so the timing measures the hardware compute
cost of a single decode step in steady state without needing to manage an advancing
position counter across warm-up resets.

```python
WARMUP   = 3
DECODE_N = 8

for _ in range(WARMUP):
    compiled_dec(input_ids_1, cache_pos)   # cache_pos = tensor([decode_len-1])

times = []
for _ in range(DECODE_N):
    t = time.time()
    compiled_dec(input_ids_1, cache_pos)
    times.append(time.time() - t)

decode_tok_s = 1.0 / statistics.mean(times[2:])   # drop 2 outliers
```

## What Gets Logged in the Bestiary

| Field | Value |
|-------|-------|
| `decode_tok_s` | Median single-step throughput (tok/s) |
| `decode_context_len` | `max_cache_len` used for the benchmark |
| `decode_note` | `"StaticCache KV cache"` or `"no KV cache — full recompute per step"` |

## Fallback: Full Recompute

If `_try_kv_decode()` fails (model doesn't support `cache_position`, config is
missing attributes, forge can't compile the decode graph), bench_decode.py falls
back to the original full-recompute approach: pad the prompt to `decode_len` and
re-run the full forward pass each step.  The result is labeled `"no KV cache —
full recompute per step"` in the bestiary so it's never confused with real decode
throughput.

## Relevant Files

| File | Purpose |
|------|---------|
| `scripts/bench_decode.py` | Main benchmark, contains `_try_kv_decode()` |
| `~/code/tt-forge-models/tools/utils.py:308` | `get_static_cache_decode_inputs()` |
| `~/code/tt-forge-models/falcon/pytorch/loader.py:320` | `load_inputs_decode()` using StaticCache |
| `forge/op/kv_cache.py` | `FillCache` / `UpdateCache` forge ops |
| `transformers/cache_utils.py:1250` | `StaticCache` class definition |
