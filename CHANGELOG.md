# Changelog

All notable changes to tt-forge-compiletron are documented here.

## [Unreleased]

### Added
- `docs/kv-cache-bench.md` — teaching companion for the StaticCache KV cache
  benchmark, explaining the two-graph pattern and why static shapes matter

---

## [1.6.0] — 2026-06-30

### Added
- **StaticCache KV cache decode benchmarking** — `bench_decode.py` now compiles
  a second forge graph for the decode step using `transformers.StaticCache`.
  The StaticCache is embedded in `KVDecodeWrapper` as a submodule so forge
  traces K/V tensors as model state and emits `FillCache`/`UpdateCache` ops.
  Falls back to full-recompute for models that don't support `cache_position`.
- `_try_kv_decode()` function — detects model dtype to avoid bfloat16/float32
  mismatches, resolves tokenizer from loader or AutoTokenizer, pre-fills cache
  on CPU before forge compilation.
- Bestiary `decode_note` field now records the method used per model
  ("StaticCache KV cache" vs "no KV cache — full recompute per step").

### Changed
- Decode results updated for all 5 stages — GPT-2 2.30→5.52 tok/s, OPT
  3.98→5.05 tok/s, Phi-2 1.48 tok/s (new), Falcon 3.30 tok/s (new),
  LLaMA-LoRA 2.86 tok/s (new), Gemma-LoRA 2.40 tok/s (new), and more.

---

## [1.5.0] — 2026-06-30

### Added
- **`scripts/bench_decode.py`** — dedicated LLM decode benchmark measuring
  TTFT, prefill tok/s, and decode tok/s for all compiled causal LMs.
  Subprocess isolation + tt-smi health check prevent hardware lockups.
- **Leaderboard columns** — TTFT, Prefill tok/s, Decode tok/s, Params (M)
  replace the old Infer p50 / Throughput columns in `docs/leaderboard.html`.
- 5 benchmark stages: Stage 1 (GPT-2, OPT), Stage 2 (Phi-2, BLOOM, CodeGen),
  Stage 3 (Falcon, Allam, LLaMA-LoRA, Gemma-LoRA), Stage 4 (Qwen 2.5,
  Phi-1 LoRA), Stage 5 (DeepCogito, DeepSeek Coder, frontier models).
- `params_m` field added to all benchmarked bestiary entries.
- `hf:` loader prefix for frontier HuggingFace models loaded without a
  tt-forge-models seed loader.

### Changed
- Bestiary `throughput_unit` relabeled from generic `tok/s` → `prefill_tok/s`
  for all 54 causal LM entries to prevent confusion with decode throughput.

---

## [1.4.0] — 2026-06-29

### Added
- **`scripts/install.sh`** — turn-key smart installer: hardware pre-check,
  hugepages, disk space, forge venv, XLA venv, mesh descriptor probe,
  tt-forge-models clone, stale-shm cleanup. Outputs color-coded summary table.
- **RAM/DRAM budget calculator** — skips models whose weights exceed available
  system RAM + per-chip DRAM; prevents OOM crashes at load time.
- **`scripts/setup-venvs.sh`** — minimal venv setup script for clean Ubuntu
  24.04 installs on Tenstorrent Blackhole hardware.
- Self-contained patches directory — tt-forge-models fixes applied at
  expedition startup without modifying upstream.
- `--ephemeral` / `--evict-failures` flags — evict HF weight cache after
  each model to reclaim disk space on small-storage machines.

### Changed
- XLA device probe verifies mesh descriptor at install time.
- `_OUT_OF_SCOPE` blocklist added for oversized LLMs (QwQ-32B, Phi-4, Flux,
  SDXL etc.) and low-quality / bot-inflated models.

---

## [1.3.0] — 2026-05-28

### Added
- **XLA backend integration** — pjrt-plugin-tt 0.9.0; JAX/Flax models now
  route to `expedition_worker_xla.py` via intelligent dispatch.
- **`--provider` flag** — run all models from a specific HuggingFace author/org.
- **`--xla-mesh N`** — multi-chip JAX dispatch with shard_map (4-chip BLOOM).
- **Intelligent dispatch** — RALLY banner, per-model backend routing, XLA
  affinity for library-tagged pytorch models.
- **Isolated overlay deps** — each model gets a disposable venv overlay
  (`--system-site-packages`) for its requirements; destroyed after compile.
- **Harness hardening (env-fix)** — dataset pre-warm, env fingerprint on
  failures, `clear_stale_env_failures()` auto-reset on version upgrades,
  `wrong_backend` removed from permafail categories.
- **IRD_LF_CACHE pre-flight** — 13 internal-server seed models fail fast
  (< 1 s) instead of downloading weights then crashing.
- **`--rerun-compiled`** — re-benchmarks already-compiled models with fresh
  inference passes to collect updated timing data.
- **Community submissions** — GitHub Action + issue parser; community data
  section on website.
- **Perf tracking** — `data/perf_history.jsonl`; `show_perf_stats.py`
  post-run summary.

### Changed
- Frontier discovery: `sort=createdAt&library=pytorch&ratio=5000` finds
  ~20–30 new models per run.
- Leaderboard: failures table removed; only compiled wins shown.

---

## [1.2.0] — 2026-05-13

### Added
- **Side quests** — idle chips compile a curated "juggle pool" (MobileNetV2,
  GhostNet, GoogleNet, etc.) while the RALLY finale assembles.
- **Full-screen wave finale** — RALLY banner fills chip-grid height; Rave
  Tapes aesthetic on model compile success.
- **EasyDel NNX path** — Type C loader for 17 EasyDel models using
  `nnx.split/merge` + Mesh.
- **Gated model preflight** — interactive gate + permafail for models
  requiring manual HuggingFace auth.
- **Config preflight** — catches custom-arch dependencies before weight
  download; permafails unknown `model_type` architectures.
- **Disk guardian** — monitors free space; offloads expedition cache to
  bonus drive when available.
- **`--no-permafail`** flag — bypass the permanent-failure gate for
  re-trying previously failed models.
- **`--pretend`** dry-run — simulate an expedition without touching hardware.
- **Dedicated leaderboard page** (`docs/leaderboard.html`) with live chip
  history, HuggingFace links for all entries.
- **MIDI model** — `skytnt/midi-model` loader + bestiary entry + curated demo slot.
- Vision-language loader (`image-text-to-text` pipeline, LLaVA-OneVision).

### Changed
- Subprocess isolation for `forge.compile()` — each model in its own process,
  `/dev/shm` cleanup on every TUI launch.
- RunState MVC refactor — single source of truth for in-flight run data.
- Bestiary concurrent-write race condition fixed; 14 missing entries recovered.

---

## [1.1.0] — 2026-05-11

### Added
- **Project website** (`docs/index.html`) — landing page with bestiary
  showcase, live `asciinema` demo cast, arch tree, and model count.
- **`scripts/record_demo.sh`** — full curated recording pipeline; `--bench`
  flag for performance capture.
- **Curated demo mode** — fixed 8-model queue (AlexNet C0, GPT-2 C1, BEiT C2,
  DenseUNet FAIL C3, BLOOM JAX 4-chip finale) with intentional failure.
- **ONNX support** — `forge.compile()` accepts `onnx.ModelProto` directly.
- **Hugepages fix** — `vm.nr_hugepages` set at launch; documented in INSTALL.md.
- **4-chip XLA data-parallel BLOOM** — genuine multi-chip finale via JAX
  `shard_map`.

### Changed
- ASCII boxes: left/bottom bars only — no right-side borders that break in
  narrow terminals.
- Terminal fixed at 220×58 for consistent recording.
- FORGE_PYTORCH_ONLY isolation for `forge.compile()` SIGSEGV.

---

## [1.0.0] — 2026-05-07

### Added
- Initial release: **tt-forge-compiletron v1.0.0**
- `expedition.py` — mass-compile TUI for Tenstorrent Blackhole hardware.
- `data/bestiary.json` — compiled/failed model registry for site + CI.
- `lib/expedition/expedition_worker.py` — forge compile + inference harness
  with error classification and permafail gate.
- **Frontier discovery** — HuggingFace search (min_downloads, min_likes,
  download-to-likes ratio guard, bot-inflation filter).
- **Seed models** — curated loaders from `tt-forge-models` for GPT-2, OPT,
  Falcon, BLOOM, Phi-2, LLaMA LoRA, Gemma LoRA, etc.
- **Docker support** — 4-way parallel compilation grid.
- `docs/PIPELINES.md`, `docs/INSTALL.md`, `docs/MULTI_CHIP.md`.
