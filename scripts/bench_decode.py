#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
bench_decode.py — Proper LLM throughput benchmarking for Tenstorrent forge.

Measures three distinct numbers for each causal LM:
  1. TTFT (time-to-first-token):  prefill latency in ms for the prompt
  2. Prefill tok/s:               prompt tokens processed per second (batch forward)
  3. Decode tok/s:                new tokens generated per second in a single-token
                                   autoregressive step, using StaticCache KV cache

Best-practice decode flow (StaticCache):
  Two compiled graphs are used — one for prefill (use_cache=False), one for decode
  (use_cache=True, StaticCache).  StaticCache has a fixed max_cache_len so forge can
  compile a single static decode graph; forge's FillCache / UpdateCache ops handle
  in-place K/V writes without dynamic tensor shapes.

  The StaticCache is embedded in KVDecodeWrapper as self.kv_cache (a submodule) rather
  than passed as a forward argument, so forge registers the K/V tensors as model state
  and the forward signature stays (input_ids: (1,1), cache_position: (1,)).

  For models that don't support StaticCache the script falls back to full-recompute
  per step (same as the original approach) and labels the result accordingly.

  See: ~/code/tt-forge-models/tools/utils.py get_static_cache_decode_inputs()
       ~/code/tt-forge-compiletron/docs/kv-cache-bench.md (this lesson)

Usage:
  python3 scripts/bench_decode.py --model opt/causal_lm/pytorch
  python3 scripts/bench_decode.py --stage 1          # small models
  python3 scripts/bench_decode.py --stage 2          # medium models
  python3 scripts/bench_decode.py --stage 3          # larger models
  python3 scripts/bench_decode.py --list-stages      # show plan
  python3 scripts/bench_decode.py --label-only       # just relabel, no hardware
"""
import sys, os, time, types, importlib, statistics, argparse, json, traceback, subprocess, tempfile

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_SITE   = "/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages"
FORGE_MODELS = os.path.expanduser("~/code/tt-forge-models")
BESTIARY    = os.path.join(REPO_ROOT, "data", "bestiary.json")

for p in [REPO_ROOT, VENV_SITE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register _forgems synthetic package so loaders can do relative imports
def _register_forgems():
    _PKG = "_forgems"
    if _PKG not in sys.modules:
        m = types.ModuleType(_PKG)
        m.__path__ = [FORGE_MODELS]
        m.__package__ = _PKG
        m.__file__ = os.path.join(FORGE_MODELS, "__init__.py")
        sys.modules[_PKG] = m

# ── Stage definitions ─────────────────────────────────────────────────────────
# Each entry: (bestiary_key, forge_models_loader_path, decode_len, notes)
# decode_len: fixed context length to compile at for the decode simulation.
STAGES = {
    1: {
        "label": "Small causal LMs (< 500M params)",
        "models": [
            # GPT-2: no attention_mask in load_inputs() output (single-input mode)
            ("gpt2/pytorch",             "gpt2.pytorch.loader",                    64, "single-input"),
            ("opt/causal_lm/pytorch",    "opt.causal_lm.pytorch.loader",           64, "list-inputs"),
            # XGLM 1.7B: compile works but per-step inference is ~68s (7GB fp32 model);
            # removed from active benchmarking — prefill=3.77 tok/s is in the bestiary.
            # ("xglm/pytorch",           "xglm.pytorch.loader",                    64, "dict-inputs"),
        ],
    },
    2: {
        "label": "Medium causal LMs (500M – 3B params)",
        "models": [
            ("phi2/causal_lm/pytorch",   "phi2.causal_lm.pytorch.loader",          64, "single-input"),
            ("bloom/pytorch",            "bloom.pytorch.loader",                    32, "dict-inputs"),
            # CodeGen: forge folds attention_mask into a constant — single-input only
            ("codegen/pytorch",          "codegen.pytorch.loader",                  32, "single-input"),
        ],
    },
    3: {
        "label": "Larger causal LMs (3B+ params or slower compile)",
        "models": [
            # Falcon/Allam/LLaMA-LoRA/Gemma-LoRA: forge folds attention_mask — single-input only
            ("falcon/pytorch",           "falcon.pytorch.loader",                   32, "single-input"),
            ("allam/causal_lm/pytorch",  "allam.causal_lm.pytorch.loader",          32, "single-input"),
            # Allam decode: TT_FATAL bank_manager OOM at 32-token ctx (7B model too large)
            ("llama_lora/causal_lm/pytorch", "llama_lora.causal_lm.pytorch.loader", 32, "single-input"),
            ("gemma_lora/pytorch",       "gemma_lora.pytorch.loader",               32, "single-input"),
        ],
    },
    4: {
        "label": "Qwen 2.5 / LoRA variants",
        "models": [
            ("qwen_2_5_coder/pytorch",         "qwen_2_5_coder.pytorch.loader",           32, "single-input"),
            ("qwen_2_5_lora/causal_lm/pytorch","qwen_2_5_lora.causal_lm.pytorch.loader",  32, "single-input"),
            ("phi1_lora/causal_lm/pytorch",    "phi1_lora.causal_lm.pytorch.loader",      32, "single-input"),
        ],
    },
    5: {
        "label": "Seed causal LMs + frontier community models",
        "models": [
            # deepcogito: 3B LLaMA-based; forge folds attention_mask — single-input
            ("deepcogito/pytorch",              "deepcogito.pytorch.loader",               32, "single-input"),
            # DeepSeek Coder 1.3B: load_inputs() returns a plain tensor (pad_inputs output)
            ("deepseek/deepseek_coder/pytorch", "deepseek.deepseek_coder.pytorch.loader",  32, "single-input"),
            # Frontier community models — loaded directly from HuggingFace
            # NovaCorp/Ultimate-RPG.System-3.2-1B: skipped — uses TokenizersBackend
            #   tokenizer class which is not available in the forge venv.
            # SpiceeChat/Bio2Tags-Lite: skipped — AutoModelForCausalLM.from_pretrained
            #   raises 'list' object has no attribute 'keys' internally (likely custom
            #   non-standard architecture that doesn't fully support AutoClass loading).
            # ("SpiceeChat/Bio2Tags-Lite", "hf:SpiceeChat/Bio2Tags-Lite", 32, "single-input"),
            # smeft-qwen-7b: 7B — likely decode OOM like Allam 7B (bank_manager)
            ("ahammad115566/smeft-qwen-7b",          "hf:ahammad115566/smeft-qwen-7b",          32, "single-input"),
            # XGLM 1.7B: skipped — per-step inference ~68s (fp32 7GB model), too slow to benchmark
            # gpt_neo/sequence_classification: skipped — classifier, not a generative decoder
        ],
    },
    6: {
        "label": "Unbenchmarked compiled text-gen frontier models",
        # 2026-06-30 run results: all 8 models failed.
        #
        # Two failure patterns observed:
        #
        # 1. AutoClass config loading: 'list' object has no attribute 'keys'
        #    Same issue as SpiceeChat/Bio2Tags-Lite (Stage 5).  Affects Llama/Qwen
        #    derivatives that have a non-standard config.json the AutoClass loader
        #    can't parse.  Not fixable without patching load_model().
        #    Affected: OvercastLab/Quark-50m-Instruct, Xerv-AI/Ada
        #
        # 2. forge FATAL: Invalid input index at runtime.cpp:956
        #    (inputIndex < program->inputs()->size())
        #    forge folds attention_mask away as a constant during prefill compilation
        #    (identical to GPT-2's behavior in Stage 1).  bench_decode uses list-inputs
        #    (forward_2 passes both input_ids + attention_mask) so the binary receives
        #    more inputs than it declared.  Fix: switch these entries to single-input.
        #    Affected: SentienceTrial1, lumasik/quark-1-248m-base,
        #              snehangshu511/gpt2-medium-instruct, OpceanAI/Yuuki-RxG-nano,
        #              itstechuse/akeno-mergedv8
        #
        # 3. Tokenizer class not found: TokenizersBackend does not exist
        #    Non-standard tokenizer class requires a package not in the forge venv.
        #    Same failure mode as NovaCorp/Ultimate-RPG.System-3.2-1B (Stage 5).
        #    Affected: abubakaraabi786/tinyllama-peft-merged
        #
        # Models left commented to avoid repeated failed runs; fix note above.
        "models": [
            # OvercastLab/Quark-50m-Instruct: FAILED — 'list' object has no attribute 'keys'
            # ("OvercastLab/Quark-50m-Instruct", "hf:OvercastLab/Quark-50m-Instruct", 64, "list-inputs"),
            # ULTR0N/SentienceTrial1: FAILED — forge FATAL invalid input index; retry with single-input
            # ("ULTR0N/SentienceTrial1",          "hf:ULTR0N/SentienceTrial1",          64, "single-input"),
            # Xerv-AI/Ada: FAILED — 'list' object has no attribute 'keys'
            # ("Xerv-AI/Ada",                     "hf:Xerv-AI/Ada",                     32, "list-inputs"),
            # lumasik/quark-1-248m-base: FAILED — forge FATAL invalid input index; retry with single-input
            # ("lumasik/quark-1-248m-base",        "hf:lumasik/quark-1-248m-base",       64, "single-input"),
            # snehangshu511/gpt2-medium-instruct: FAILED — forge FATAL invalid input index; retry with single-input
            # ("snehangshu511/gpt2-medium-instruct", "hf:snehangshu511/gpt2-medium-instruct", 64, "single-input"),
            # abubakaraabi786/tinyllama-peft-merged: FAILED — TokenizersBackend not found
            # ("abubakaraabi786/tinyllama-peft-merged", "hf:abubakaraabi786/tinyllama-peft-merged", 32, "list-inputs"),
            # OpceanAI/Yuuki-RxG-nano: FAILED — forge FATAL invalid input index; retry with single-input
            # ("OpceanAI/Yuuki-RxG-nano",          "hf:OpceanAI/Yuuki-RxG-nano",         64, "single-input"),
            # itstechuse/akeno-mergedv8: FAILED — forge FATAL invalid input index; retry with single-input
            # ("itstechuse/akeno-mergedv8",         "hf:itstechuse/akeno-mergedv8",       32, "single-input"),
        ],
    },
    7: {
        "label": "Stage 6 single-input retries (forge folds attention_mask)",
        # Retry the 5 models that failed stage 6 with forge FATAL invalid input index.
        # Root cause: forge folds attention_mask to a constant at compile time, so the
        # compiled binary only accepts input_ids.  list-inputs passes both → index OOB.
        # These use single-input mode (CausalLMWrapper.forward_1 — input_ids only).
        "models": [
            ("ULTR0N/SentienceTrial1",               "hf:ULTR0N/SentienceTrial1",               64, "single-input"),
            ("lumasik/quark-1-248m-base",             "hf:lumasik/quark-1-248m-base",            64, "single-input"),
            ("snehangshu511/gpt2-medium-instruct",    "hf:snehangshu511/gpt2-medium-instruct",   64, "single-input"),
            ("OpceanAI/Yuuki-RxG-nano",               "hf:OpceanAI/Yuuki-RxG-nano",              64, "single-input"),
            ("itstechuse/akeno-mergedv8",             "hf:itstechuse/akeno-mergedv8",            32, "single-input"),
        ],
    },
}

# ── Generic causal LM wrapper ─────────────────────────────────────────────────
def _make_wrapper(torch):
    class CausalLMWrapper(torch.nn.Module):
        """Returns only logits — strips DynamicCache / CausalLMOutput wrappers.

        Uses return_dict=False so the model returns a plain tuple; logits are
        always out[0].  This avoids conflicts where config.return_dict=False
        overrides a return_dict=True kwarg (seen in some frontier Qwen models).
        """
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward_2(self, input_ids, attention_mask):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=False, return_dict=False)
            return out[0] if isinstance(out, (tuple, list)) else out.logits

        def forward_1(self, input_ids):
            out = self.model(input_ids=input_ids,
                             use_cache=False, return_dict=False)
            return out[0] if isinstance(out, (tuple, list)) else out.logits

        forward = forward_2  # default; may be swapped to forward_1 below

    class Single(torch.nn.Module):
        """Single-input variant for models where forge folds attention_mask."""
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids):
            out = self.model(input_ids=input_ids, use_cache=False, return_dict=False)
            return out[0] if isinstance(out, (tuple, list)) else out.logits

    return CausalLMWrapper, Single


# ── StaticCache KV cache decode ───────────────────────────────────────────────
def _try_kv_decode(model, loader_inst, tokenizer_override, decode_len, torch, forge, log):
    """Compile a StaticCache-backed single-token decode graph.

    Two compiled graphs are the forge best practice for causal LMs:
      • Prefill graph (use_cache=False): processes full prompt, returns logits.
      • Decode graph (use_cache=True, StaticCache): single-token step, updates
        K/V cache in-place via forge FillCache/UpdateCache ops.

    StaticCache works with forge because it has a fixed max_cache_len — no
    dynamic tensor shapes.  The cache is embedded in KVDecodeWrapper as
    self.kv_cache (a torch.nn.Module submodule) so forge registers the K/V
    tensors as compiled model state, not as dynamic forward arguments.

    Args:
        model:             Raw PyTorch causal LM (not yet forge-compiled).
        loader_inst:       ModelLoader instance (for .tokenizer), or None for hf: models.
        tokenizer_override: Pre-built tokenizer for hf: frontier models.
        decode_len:        max_cache_len for the StaticCache.
        torch/forge/log:   Injected so the caller controls imports.

    Returns:
        (compiled_dec, input_ids_1tok, cache_position_tensor) on success.
        None if the model doesn't support StaticCache or forge can't compile it.
    """
    # ── Import helpers ────────────────────────────────────────────────────
    try:
        import transformers
        import importlib as _il
        if FORGE_MODELS not in sys.path:
            sys.path.insert(0, FORGE_MODELS)
        _tools = _il.import_module("tools.utils")
        _get_dec = getattr(_tools, "get_static_cache_decode_inputs", None)
        if _get_dec is None:
            log("decode KV: tools.utils.get_static_cache_decode_inputs not found")
            return None
    except Exception as e:
        log(f"decode KV: import failed: {e}")
        return None

    # ── Resolve tokenizer ─────────────────────────────────────────────────
    # Loaders expose self.tokenizer after load_inputs() has been called.
    # For hf: frontier models the tokenizer was already built by the caller.
    try:
        if tokenizer_override is not None:
            tok = tokenizer_override
        else:
            tok = getattr(loader_inst, "tokenizer", None)
            if tok is None:
                # Fall back to AutoTokenizer using the config's HF model ID
                name = getattr(getattr(model, "config", None), "_name_or_path", None)
                if not name:
                    log("decode KV: no tokenizer and no config._name_or_path")
                    return None
                tok = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception as e:
        log(f"decode KV: tokenizer load failed: {e}")
        return None

    # ── Pre-fill StaticCache on CPU ───────────────────────────────────────
    # get_static_cache_decode_inputs() runs a CPU forward pass to fill the
    # cache with realistic K/V content, then returns the single-token decode
    # inputs dict: {input_ids (1,1), past_key_values (StaticCache),
    #               cache_position (1,), use_cache: True}.
    # Use the model's actual dtype for the cache to avoid attention dtype
    # mismatches (e.g. float32 model with bfloat16 StaticCache default).
    try:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        dec_in = _get_dec(
            tokenizer=tok,
            config=model.config,
            model=model,
            batch_size=1,
            max_cache_len=decode_len,
            device="cpu",
            dtype=model_dtype,
        )
    except Exception as e:
        log(f"decode KV: pre-fill StaticCache failed: {e}")
        return None

    static_cache = dec_in["past_key_values"]   # StaticCache (torch.nn.Module)
    input_ids_1  = dec_in["input_ids"]          # (1, 1) long tensor
    cache_pos    = dec_in["cache_position"]      # (1,) long tensor = [decode_len-1]

    # ── KV decode wrapper ─────────────────────────────────────────────────
    # StaticCache is registered as self.kv_cache — a torch.nn.Module submodule.
    # forge traces its key_cache / value_cache tensors as part of the compiled
    # graph and replaces cache.update() calls with FillCache / UpdateCache ops.
    # Forward signature: (input_ids: (1,1), cache_position: (1,)) — fixed shapes.
    class KVDecodeWrapper(torch.nn.Module):
        def __init__(self, m, kvc):
            super().__init__()
            self.model = m
            self.kv_cache = kvc  # StaticCache submodule; K/V tensors become graph state

        def forward(self, input_ids, cache_position):
            out = self.model(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=False,
            )
            return out[0] if isinstance(out, (tuple, list)) else out.logits

    # ── forge.compile decode graph ────────────────────────────────────────
    # Disable grad on all parameters before compile.  During forge's internal
    # TVM trace the forward pass runs with real tensors; with use_cache=True
    # the attention K/V projections write into the StaticCache, and if those
    # intermediate tensors require grad the TVM constant-folding pass calls
    # .numpy() on them and raises RuntimeError.  Prefill works fine (use_cache
    # =False never writes to the cache), so only the decode graph needs this.
    try:
        kv_wrapper = KVDecodeWrapper(model, static_cache)
        for p in kv_wrapper.parameters():
            p.requires_grad_(False)
        log(f"compiling decode KV cache (StaticCache, ctx={decode_len}) ...")
        compiled_dec = forge.compile(kv_wrapper, sample_inputs=[input_ids_1, cache_pos])
        log("decode KV cache compile OK")
        return compiled_dec, input_ids_1, cache_pos
    except Exception as e:
        log(f"decode KV cache compile failed (will use full-recompute fallback): {e}")
        return None


# ── Input normalisation ───────────────────────────────────────────────────────
def _norm_inputs(raw, torch, input_mode):
    """Return (input_ids, attention_mask) from loader output (list, dict, or BatchEncoding).

    BatchEncoding is a UserDict subclass — fails isinstance(raw, dict) — so we
    use hasattr(raw, 'keys') to catch all mapping types.

    Plain 2D tensors (e.g. deepseek_coder pad_inputs return value) are returned
    directly — raw[0] would give a 1D tensor and break shape[1] lookups.
    """
    # Plain tensor — already (batch, seq): return as-is.
    if hasattr(raw, 'shape') and not hasattr(raw, 'keys'):
        return raw, None
    is_mapping = hasattr(raw, 'keys')
    if input_mode == "single-input":
        if is_mapping:
            return raw["input_ids"], None
        return raw[0], None
    if is_mapping:
        return raw["input_ids"], raw.get("attention_mask")
    # plain list
    return raw[0], raw[1] if len(raw) > 1 else None


def _pad_inputs(input_ids, attn_mask, decode_len, torch):
    """Pad (or truncate) input to decode_len for fixed-shape compilation."""
    cur = input_ids.shape[1]
    if cur == decode_len:
        return input_ids, attn_mask
    if cur > decode_len:
        # Use clone() to get a contiguous tensor — a plain slice keeps the original stride
        ids  = input_ids[:, :decode_len].clone()
        mask = attn_mask[:, :decode_len].clone() if attn_mask is not None else None
        return ids, mask
    pad = decode_len - cur
    pad_ids = torch.zeros((1, pad), dtype=input_ids.dtype)
    ids = torch.cat([input_ids, pad_ids], dim=1)
    if attn_mask is not None:
        pad_mask = torch.zeros((1, pad), dtype=attn_mask.dtype)
        mask = torch.cat([attn_mask, pad_mask], dim=1)
    else:
        mask = None
    return ids, mask


# ── Core benchmark function ───────────────────────────────────────────────────
WARMUP   = 3
BENCH    = 5
DECODE_N = 8   # decode steps to simulate

def _run_bench(model_key, loader_dotpath, decode_len, input_mode,
               verbose=True) -> dict:
    """
    Compile and benchmark one causal LM.

    Returns a result dict with keys:
      prefill_tok_s, ttft_ms, decode_tok_s, decode_context_len,
      compile_s, error (if failed)
    """
    import torch   # import inside to avoid loading before venv is on path
    import forge

    result = {
        "model_key":          model_key,
        "prefill_tok_s":      None,
        "ttft_ms":            None,
        "decode_tok_s":       None,
        "decode_context_len": decode_len,
        "decode_note":        None,
        "compile_s":          None,
        "error":              None,
    }

    def log(msg):
        if verbose:
            print(f"  {msg}", flush=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    # Track loader_inst and tokenizer_hf separately so _try_kv_decode can
    # access the tokenizer without re-loading it.
    loader_inst     = None   # ModelLoader instance (seed models)
    tokenizer_hf    = None   # Pre-built tokenizer (hf: frontier models)

    try:
        if loader_dotpath.startswith("hf:"):
            # Frontier model — load directly via AutoModelForCausalLM.
            # These were compiled by the dynamic loader in the expedition;
            # bench_decode re-loads them the same way and tokenizes a fixed prompt.
            import transformers
            hf_model_id = loader_dotpath[3:]
            tokenizer_hf = transformers.AutoTokenizer.from_pretrained(
                hf_model_id, trust_remote_code=True
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                hf_model_id, trust_remote_code=True
            )
            # Do NOT set config.return_dict = False — wrappers use return_dict=False
            # explicitly in their forward() calls so the kwarg always wins.
            tok_out = tokenizer_hf("The quick brown fox jumps over the lazy dog",
                                   return_tensors="pt")
            # Some tokenizers return a list or tensor rather than BatchEncoding;
            # normalise to a dict with input_ids so _norm_inputs can handle it.
            if hasattr(tok_out, 'keys'):
                raw_inputs = tok_out
            elif isinstance(tok_out, (list, tuple)):
                raw_inputs = {"input_ids": tok_out[0]}
            else:
                raw_inputs = {"input_ids": tok_out}
        else:
            _register_forgems()
            mod = importlib.import_module(f"_forgems.{loader_dotpath}")
            loader_inst = mod.ModelLoader()
            model = loader_inst.load_model()
            raw_inputs = loader_inst.load_inputs()
            # loader_inst.tokenizer is set by load_inputs() — available for KV decode
        model.eval()
    except Exception as e:
        result["error"] = f"load_model failed: {e}"
        return result

    input_ids_orig, attn_mask_orig = _norm_inputs(raw_inputs, torch, input_mode)
    prompt_len = input_ids_orig.shape[1]
    log(f"loaded — prompt_len={prompt_len}  params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    CausalLMWrapper, Single = _make_wrapper(torch)

    # ── 2. Prefill benchmark (original prompt length) ──────────────────────
    try:
        if input_mode == "single-input":
            wrapper_pf = Single(model)
            sample_pf  = [input_ids_orig]
        else:
            wrapper_pf = CausalLMWrapper(model)
            sample_pf  = [input_ids_orig, attn_mask_orig] if attn_mask_orig is not None else [input_ids_orig]

        log(f"compiling prefill at len={prompt_len} ...")
        t0 = time.time()
        compiled_pf = forge.compile(wrapper_pf, sample_inputs=sample_pf)
        compile_s = time.time() - t0
        result["compile_s"] = round(compile_s, 1)
        log(f"compile done in {compile_s:.1f}s")

        # warmup
        for _ in range(WARMUP):
            compiled_pf(*sample_pf)

        # TTFT = single forward pass latency = prefill latency
        times_pf = []
        for _ in range(BENCH):
            t = time.time()
            out = compiled_pf(*sample_pf)
            times_pf.append(time.time() - t)

        p50_pf = statistics.median(times_pf)
        result["ttft_ms"]       = round(p50_pf * 1000, 1)
        result["prefill_tok_s"] = round(prompt_len / p50_pf, 1)
        log(f"prefill: {p50_pf*1000:.0f}ms  →  {result['prefill_tok_s']:.1f} tok/s")

    except Exception as e:
        result["error"] = f"prefill failed: {e}"
        log(f"PREFILL FAILED: {e}")
        return result

    # ── 3. Decode benchmark ────────────────────────────────────────────────
    # Best-practice path: StaticCache KV cache (two compiled graphs).
    #   - _try_kv_decode() pre-fills a StaticCache on CPU, wraps the model
    #     with the cache as a submodule, and calls forge.compile() on the
    #     single-token decode graph.
    #   - Each timed step calls compiled_dec(input_ids_1tok, cache_pos) where
    #     cache_pos is fixed at max_cache_len-1 (we benchmark step speed, not
    #     generation accuracy; overwriting the same slot is fine for timing).
    #   - forge handles in-place cache writes via FillCache/UpdateCache ops.
    #   - NOTE: prompt_len has no bearing on the KV decode benchmark — the
    #     cache is pre-filled independently using decode_len as max_cache_len.
    #     Always attempt KV decode regardless of prompt_len vs decode_len.
    #
    # Fallback (when KV compile fails):
    #   If prompt_len >= decode_len: derive decode cost from prefill latency
    #   (one full forward pass of prompt_len tokens per step).
    #   Otherwise: compile at decode_len with use_cache=False and measure.
    #   Both are labeled "no KV cache — full recompute" in the bestiary.
    try:
        # ── 3a. Try StaticCache KV cache decode (always — independent of prompt_len)
        kv = _try_kv_decode(
            model=model,
            loader_inst=loader_inst,
            tokenizer_override=tokenizer_hf,
            decode_len=decode_len,
            torch=torch, forge=forge, log=log,
        )
        if kv is not None:
            compiled_dec, input_ids_1, cache_pos = kv

            for _ in range(WARMUP):
                compiled_dec(input_ids_1, cache_pos)

            times_dc = []
            for _ in range(DECODE_N):
                t = time.time()
                compiled_dec(input_ids_1, cache_pos)
                times_dc.append(time.time() - t)

            mean_step = statistics.mean(times_dc[2:])
            result["decode_tok_s"]       = round(1.0 / mean_step, 2)
            result["decode_context_len"] = decode_len
            result["decode_note"]        = "StaticCache KV cache"
            log(f"decode (KV cache, StaticCache): "
                f"{mean_step*1000:.0f}ms/step → {result['decode_tok_s']:.2f} tok/s  (ctx={decode_len})")

        else:
            # ── 3b. KV decode not available — fall back ───────────────────
            if prompt_len >= decode_len:
                # Derive from prefill — no second compile needed
                ttft_s = result["ttft_ms"] / 1000.0
                result["decode_tok_s"]         = round(1.0 / ttft_s, 2)
                result["decode_context_len"]   = prompt_len
                result["decode_note"]          = "no KV cache — full recompute (derived from prefill)"
                log(f"decode (no KV cache, derived from prefill, ctx={prompt_len}): "
                    f"{result['decode_tok_s']:.2f} tok/s")
            else:
                ids_dec, mask_dec = _pad_inputs(input_ids_orig, attn_mask_orig, decode_len, torch)

                if input_mode == "single-input":
                    wrapper_dc = Single(model)
                    sample_dc  = [ids_dec]
                else:
                    wrapper_dc = CausalLMWrapper(model)
                    sample_dc  = [ids_dec, mask_dec] if mask_dec is not None else [ids_dec]

                log(f"compiling decode (full-recompute fallback) at len={decode_len} ...")
                t0 = time.time()
                compiled_dc = forge.compile(wrapper_dc, sample_inputs=sample_dc)
                log(f"decode fallback compile done in {time.time()-t0:.1f}s")

                for _ in range(WARMUP):
                    compiled_dc(*sample_dc)

                times_dc = []
                for _ in range(DECODE_N):
                    t = time.time()
                    compiled_dc(*sample_dc)
                    times_dc.append(time.time() - t)

                mean_step = statistics.mean(times_dc[2:])
                result["decode_tok_s"]       = round(1.0 / mean_step, 2)
                result["decode_context_len"] = decode_len
                result["decode_note"]        = "no KV cache — full recompute per step"
                log(f"decode (no KV cache fallback): "
                    f"{mean_step*1000:.0f}ms/step → {result['decode_tok_s']:.2f} tok/s  (ctx={decode_len})")

    except Exception as e:
        log(f"decode bench failed: {e}")
        result["decode_tok_s"] = None
        # Not fatal — prefill succeeded

    return result


# ── Bestiary helpers ──────────────────────────────────────────────────────────
def load_bestiary():
    with open(BESTIARY) as f:
        return json.load(f)

def save_bestiary(b):
    with open(BESTIARY, "w") as f:
        json.dump(b, f, indent=2)
    print(f"  bestiary saved ({len(b['compiled'])} compiled, {len(b['failed'])} failed)")


def label_prefill(b):
    """
    Relabel all existing causal LM throughput entries so best_throughput is
    clearly understood as prefill throughput.  Adds prefill_tok_s field and
    updates throughput_unit; does NOT remove best_throughput (for back-compat).
    """
    CAUSAL_TASKS = {"text-generation", "nlp_causal_lm", "nlp_text_cls",
                    "nlp_masked_lm", "fill-mask"}
    changed = 0
    for key, entry in b["compiled"].items():
        if "best_throughput" not in entry:
            continue
        unit = entry.get("throughput_unit", "")
        if "prefill" in unit:
            continue  # already labeled
        task = entry.get("task", "")
        if task in CAUSAL_TASKS or task == "text-generation":
            entry["throughput_unit"] = "prefill_tok/s"
            if "prefill_tok_s" not in entry:
                entry["prefill_tok_s"] = entry["best_throughput"]
        changed += 1
    print(f"  labeled {changed} entries as prefill")
    return b


def update_entry(b, model_key, res):
    """Write decode benchmark results back into the compiled entry."""
    entry = b["compiled"].get(model_key)
    if entry is None:
        print(f"  WARNING: {model_key} not in compiled — skipping bestiary update")
        return
    if res.get("prefill_tok_s"):
        entry["prefill_tok_s"]    = res["prefill_tok_s"]
        entry["throughput_unit"]  = "prefill_tok/s"
        entry["best_throughput"]  = res["prefill_tok_s"]
        entry["ttft_ms"]          = res["ttft_ms"]
    if res.get("decode_tok_s"):
        entry["decode_tok_s"]       = res["decode_tok_s"]
        entry["decode_context_len"] = res["decode_context_len"]
        # Preserve the decode method note from the benchmark (KV cache vs full recompute)
        entry["decode_note"]        = res.get("decode_note", "no KV cache — full recompute per step")
    if res.get("compile_s"):
        entry["best_compile_s"] = res["compile_s"]


# ── Hardware safety layer ─────────────────────────────────────────────────────
# Three defensive layers run before every model subprocess:
#   1. Stale /dev/shm cleanup  — forge leaves shared-memory segments behind on
#      crash; the next forge init can re-attach them and wedge the driver.
#   2. tt-smi health check     — verify all 4 chips are alive and not thermal-
#      throttling before we hand them a model.  Skip (with a warning) rather
#      than hard-abort so a single flaky chip doesn't stop the whole stage.
#   3. Subprocess isolation    — every model runs in its own Python process so
#      forge's device handle is fully released (and the kernel driver ref-
#      counted back to zero) before the next model opens it.

SHM_GLOB = "sm_segment.tt-quietbox.*.0"  # forge shared-memory naming pattern
TT_SMI   = "tt-smi"
# ASIC_TEMPERATURE encoding: bits[23:16] = integer °C (empirically verified)
TEMP_SHIFT      = 16
TEMP_WARN_C     = 75   # log a warning above this
TEMP_ABORT_C    = 85   # refuse to start a new model above this

def _cleanup_stale_shm():
    """Delete leftover forge shared-memory segments from crashed runs."""
    import glob
    segments = glob.glob(f"/dev/shm/{SHM_GLOB}")
    if segments:
        print(f"  [preflight] cleaning {len(segments)} stale shm segment(s)", flush=True)
        for seg in segments:
            try:
                os.unlink(seg)
            except OSError:
                pass


def _try_reset_devices() -> bool:
    """Attempt tt-smi -r all to recover hung chips. Returns True if reset ran."""
    try:
        proc = subprocess.run([TT_SMI, "-r", "all"], capture_output=True, text=True, timeout=60)
        return proc.returncode == 0
    except Exception:
        return False


def _hw_health_check(auto_reset: bool = True) -> tuple[bool, str]:
    """Run tt-smi -s and verify all chips are alive and cool.

    If auto_reset=True (default) and tt-smi exits non-zero (hung chip),
    attempts tt-smi -r all once before re-checking.

    Returns (ok, message).  ok=False means skip the model.
    """
    def _query() -> tuple[bool, str, dict | None]:
        try:
            proc = subprocess.run(
                [TT_SMI, "-s"], capture_output=True, text=True, timeout=15
            )
            if proc.returncode != 0:
                return False, f"tt-smi exited {proc.returncode}: {proc.stderr.strip()[:120]}", None
            return True, "", json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            return False, "tt-smi timed out — driver may be hung", None
        except Exception as e:
            return False, f"tt-smi error: {e}", None

    ok, err, data = _query()
    if not ok and auto_reset:
        print(f"  [preflight] tt-smi reported error ({err[:60]}) — attempting tt-smi -r all ...", flush=True)
        if _try_reset_devices():
            ok, err, data = _query()
            if ok:
                print("  [preflight] reset succeeded, re-checking ...", flush=True)
        if not ok:
            return False, f"tt-smi still unhealthy after reset: {err}"

    if not ok:
        return False, err

    chips = data.get("device_info", [])
    if not chips:
        return False, "tt-smi returned 0 chips — hardware not visible"

    warnings = []
    for i, chip in enumerate(chips):
        telem = chip.get("smbus_telem", {})

        # Check ARCCLK (ARC firmware running); 0 means chip is dead
        arcclk_raw = telem.get("ARCCLK")
        try:
            arcclk = int(arcclk_raw, 16) if isinstance(arcclk_raw, str) else arcclk_raw
        except (ValueError, TypeError):
            arcclk = None
        if arcclk is not None and arcclk == 0:
            return False, f"chip {i} ARCCLK=0 — ARC firmware not running"

        # Decode ASIC temperature: bits[23:16] — value arrives as hex string
        raw_temp = telem.get("ASIC_TEMPERATURE")
        if raw_temp is not None:
            try:
                raw_int = int(raw_temp, 16) if isinstance(raw_temp, str) else raw_temp
                temp_c = (raw_int >> TEMP_SHIFT) & 0xFF
                if temp_c >= TEMP_ABORT_C:
                    return False, f"chip {i} ASIC temp {temp_c}°C ≥ abort threshold {TEMP_ABORT_C}°C"
                if temp_c >= TEMP_WARN_C:
                    warnings.append(f"chip {i} temp {temp_c}°C (warm)")
            except (ValueError, TypeError):
                pass  # non-numeric temp value — skip check

        # DDR_STATUS 0x55555555 = all links OK — arrives as hex string
        ddr_raw = telem.get("DDR_STATUS")
        try:
            ddr = int(ddr_raw, 16) if isinstance(ddr_raw, str) else ddr_raw
        except (ValueError, TypeError):
            ddr = None
        if ddr is not None and ddr not in (0x55555555, None):
            warnings.append(f"chip {i} DDR_STATUS={hex(ddr)} (non-nominal)")

    msg = f"{len(chips)} chips OK"
    if warnings:
        msg += "  WARNINGS: " + ", ".join(warnings)
    return True, msg


# ── Subprocess isolation ──────────────────────────────────────────────────────
# Each model benchmark runs in its own subprocess so forge's device state
# (TT chip allocations, shared memory segments) is fully released between
# models.  Loading a 1.7B model after two smaller ones in the same process
# causes forge device state accumulation that can hard-lock the machine.

def _run_bench_isolated(model_key, loader_dotpath, decode_len, input_mode,
                        verbose=True) -> dict:
    """Spawn bench in a subprocess; return parsed result dict.

    Pre-flight order for every model:
      1. Clean stale forge /dev/shm segments from prior crashes
      2. tt-smi hardware health check — skip model if chips unhealthy
      3. Spawn isolated subprocess (forge device context fully released on exit)
    """
    # ── Pre-flight ─────────────────────────────────────────────────────────
    _cleanup_stale_shm()

    hw_ok, hw_msg = _hw_health_check()
    print(f"  [preflight] hw: {hw_msg}", flush=True)
    if not hw_ok:
        return {
            "model_key":           model_key,
            "prefill_tok_s":       None,
            "ttft_ms":             None,
            "decode_tok_s":        None,
            "decode_context_len":  decode_len,
            "compile_s":           None,
            "error":               f"hw preflight failed: {hw_msg}",
        }

    # ── Spawn isolated worker ───────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="bench_result_", delete=False
    ) as tf:
        result_path = tf.name

    worker_args = json.dumps({
        "model_key":       model_key,
        "loader_dotpath":  loader_dotpath,
        "decode_len":      decode_len,
        "input_mode":      input_mode,
        "verbose":         verbose,
        "result_path":     result_path,
    })

    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", worker_args]
    try:
        proc = subprocess.run(cmd, timeout=900)  # 15 min hard limit per model
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            os.unlink(result_path)
            return result
        # Subprocess exited without writing a result (crash / OOM / SIGSEGV)
        rc = proc.returncode
        return {
            "model_key":           model_key,
            "prefill_tok_s":       None,
            "ttft_ms":             None,
            "decode_tok_s":        None,
            "decode_context_len":  decode_len,
            "compile_s":           None,
            "error":               f"subprocess exited rc={rc} without result",
        }
    except subprocess.TimeoutExpired:
        # Subprocess is hung (possible hardware lockup) — kill it
        print(f"  [WARNING] subprocess timed out for {model_key} — killing", flush=True)
        return {
            "model_key":           model_key,
            "prefill_tok_s":       None,
            "ttft_ms":             None,
            "decode_tok_s":        None,
            "decode_context_len":  decode_len,
            "compile_s":           None,
            "error":               "subprocess timed out (>900s) — possible hw lockup",
        }
    except Exception as e:
        return {
            "model_key":           model_key,
            "prefill_tok_s":       None,
            "ttft_ms":             None,
            "decode_tok_s":        None,
            "decode_context_len":  decode_len,
            "compile_s":           None,
            "error":               f"subprocess error: {e}",
        }
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)


def _worker_entry(worker_args_json):
    """Entry point when running as a subprocess worker.

    Runs _run_bench() for a single model and writes the JSON result to the
    path embedded in the args.  Called via --_worker flag in main().
    """
    kw = json.loads(worker_args_json)
    result_path = kw.pop("result_path")
    result = _run_bench(**kw)
    with open(result_path, "w") as f:
        json.dump(result, f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Subprocess worker mode — spawned by _run_bench_isolated(), not called by users
    if len(sys.argv) >= 3 and sys.argv[1] == "--_worker":
        _worker_entry(sys.argv[2])
        return

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",       action="append", help="Bestiary key (repeatable): --model foo --model bar")
    p.add_argument("--stage",       type=int, choices=STAGES.keys(), help="Run a predefined stage")
    p.add_argument("--decode-len",  type=int, default=None, help="Override decode context length")
    p.add_argument("--list-stages", action="store_true", help="Show stage plan and exit")
    p.add_argument("--label-only",  action="store_true", help="Relabel prefill only, no hardware run")
    p.add_argument("--preflight",   action="store_true", help="Hardware health check only (no forge)")
    args = p.parse_args()

    if args.preflight:
        print("Running hardware pre-flight checks...")
        _cleanup_stale_shm()
        ok, msg = _hw_health_check()
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {msg}")
        sys.exit(0 if ok else 1)

    if args.list_stages:
        for s, info in STAGES.items():
            print(f"\nStage {s}: {info['label']}")
            for key, loader, dlen, mode in info["models"]:
                print(f"  {key}  (decode_len={dlen}, input_mode={mode})")
        return

    b = load_bestiary()

    # Always relabel first (no hardware needed)
    b = label_prefill(b)
    save_bestiary(b)

    if args.label_only:
        print("Label-only mode — done.")
        return

    # Collect models to run
    targets = []
    if args.stage:
        info = STAGES[args.stage]
        print(f"\nStage {args.stage}: {info['label']}")
        targets = [(k, l, args.decode_len or d, m) for k, l, d, m in info["models"]]
    elif args.model:
        # Find each requested model key in stages
        stage_map = {k: (k, l, d, m) for sinfo in STAGES.values() for k, l, d, m in sinfo["models"]}
        for req in args.model:
            if req in stage_map:
                k, l, d, m = stage_map[req]
                targets.append((k, l, args.decode_len or d, m))
            else:
                print(f"Model '{req}' not in stage list — skipping")
        if not targets:
            return
    else:
        p.print_help()
        return

    # Run each model
    results_summary = []
    for model_key, loader_dotpath, decode_len, input_mode in targets:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_key}")
        print(f"  loader: {loader_dotpath}  decode_len: {decode_len}  mode: {input_mode}")
        print(f"{'='*60}")

        res = _run_bench_isolated(model_key, loader_dotpath, decode_len, input_mode)

        if res["error"]:
            print(f"  ✗ FAILED: {res['error']}")
            results_summary.append((model_key, "FAILED", res["error"]))
        else:
            print(f"  ✓ compile={res['compile_s']}s  "
                  f"TTFT={res['ttft_ms']}ms  "
                  f"prefill={res['prefill_tok_s']} tok/s  "
                  f"decode={res['decode_tok_s']} tok/s (ctx={decode_len})")
            update_entry(b, model_key, res)
            results_summary.append((model_key, "OK", res))

        # Save after each model (don't lose results if later model crashes)
        save_bestiary(b)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':45s} {'Status':8s} {'TTFT':8s} {'Prefill':10s} {'Decode':10s}")
    print("-" * 85)
    for model_key, status, data in results_summary:
        if status == "FAILED":
            print(f"{model_key:45s} {'FAILED':8s}  {data}")
        else:
            print(f"{model_key:45s} {'OK':8s} "
                  f"{data['ttft_ms']:6.0f}ms "
                  f"{data['prefill_tok_s']:8.1f} t/s "
                  f"{(data['decode_tok_s'] or 0):8.2f} t/s")


if __name__ == "__main__":
    main()
