# TT-Forge Compiletron: the expedition

> *Every run is a hunt. The HuggingFace frontier is vast. The silicon does not forgive.*

Compiletron is a **roguelike model-compilation game** for Tenstorrent hardware.
Each expedition discovers AI models from the wild, throws them at every available
chip in parallel, and scores the results by rarity and novelty. Victories are
immortalized in a persistent **bestiary**. Failures are catalogued, classified,
and eventually avenged when the forge grows stronger.

Two compilation backends: **tt-forge** (PyTorch via forge) and **tt-xla**
(JAX/Flax via PJRT plugin) — selectable per-chip or in mixed mode, with automatic
per-model routing that learns from your bestiary's crash history.

![TT-Forge Compiletron — live expedition demo](docs/demo.gif)

**Tested on:** 4× P300C Blackhole chips

---

## Quick start

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron
pip install -r requirements.txt

# Activate tt-forge backend
source ~/tt-forge-fe/env/activate

# Launch TUI (recommended) — auto-starts after 4 seconds
python3 expedition.py run --tui

# Or CLI, 4 chips, 20 models
python3 expedition.py run --chips 4 --limit 20
```

### XLA backend (JAX/PJRT)

```bash
# One-time setup
python3 -m venv xla-venv
xla-venv/bin/pip install pjrt-plugin-tt jax==0.7.1 jaxlib==0.7.1 \
    flax==0.8.5 "transformers<5.0" torch pyfiglet \
    --index-url https://pypi.tenstorrent.com/simple/

# Run with XLA backend
python3 expedition.py run --tui --backend xla
```

---

## The TUI

`expedition.py run --tui` opens a 3-screen Textual app:

```
╔══════════════════════════════════════════════════════════════
║  EXPEDITION #008 SETUP

  Seeds: tt-forge-models + HuggingFace frontier
  Backend: forge  [cycle with 5]
  Chips:   4      Limit: 20

  [Enter] Start   [Q] Quit
  ● ENTER to start  (auto in 3s)
╚══════════════════════════════════════════════════════════════
```

**Setup screen** — configure chips, limit, backend (forge / xla / mixed),
and source filters. Press Enter to start immediately, or wait 4 seconds for
auto-start. Pass `--auto-quit N` to also exit automatically N seconds after the
summary screen, enabling fully unattended recording.

**Run screen** — one panel per chip, live event log, scrolling pyfiglet ASCII
banners of each model name, real-time scores, and First Voice inference output.

**Summary screen** — Field Report aesthetic. NATO codenames (ALPHA/BRAVO/CHARLIE/DELTA)
ranked by points. MISSION SUMMARY with classification badge (OUTSTANDING/COMPILED/PARTIAL/CRITICAL),
NEW INTELLIGENCE for first-ever models (INTERCEPT for first voice, ARTIFACT for tensor stats),
compact CATALOGUED list for other successes, and TARGETS AT LARGE for failures. All-time
bestiary totals at the bottom. Auto-advances from the wave finale after 2 seconds.

---

## CLI reference

```bash
# Run modes
python3 expedition.py run --tui                    # interactive TUI, 4s auto-start
python3 expedition.py run --chips 4 --limit 20     # CLI, 4 chips, 20 models
python3 expedition.py run --backend xla            # JAX/PJRT backend
python3 expedition.py run --backend mixed          # even chips=forge, odd=xla
python3 expedition.py run --seed-only              # tt-forge-models zoo only
python3 expedition.py run --frontier-only          # HuggingFace frontier only
python3 expedition.py run --staples                # re-run proven seed models (skip perm-fail gate)

# Discovery filters
python3 expedition.py run --min-downloads 1000     # skip obscure models
python3 expedition.py run --min-likes 5            # skip experiment dumps
python3 expedition.py run --max-model-params 7     # single-chip sweet-spot

# Download controls
python3 expedition.py run --max-cache-gb 150       # cap HF cache at 150 GB
python3 expedition.py run --session-download-max 60 # limit this run to 60 GB
python3 expedition.py run --no-predownload         # skip pre-fetch, start faster

# Hardware
python3 expedition.py run --monitor                # add tt-smi pane
```

---

## How it works

```
expedition.py               CLI + queue builder
expedition_tui.py           Textual TUI (Setup / Run / Summary screens)
lib/expedition/
  expedition_worker.py      per-chip forge worker (PyTorch / tt-forge)
  expedition_worker_xla.py  per-chip XLA worker  (JAX / PJRT)
  bestiary.py               compiled-model database (data/bestiary.json)
  scorer.py                 rarity × newness → points
  decoder.py                output → human-readable First Voice text
  sampler.py                themed inference samples per task type
  hud.py                    per-chip stats tracker
  notes.py                  run journal (data/expeditions/)
  hf_discover.py            live HuggingFace frontier discovery
  router.py                 per-model backend dispatch (auto mode)
```

**Queue building** — each run scans the tt-forge-models JAX/PyTorch zoo and
the HuggingFace Hub for recently-created models. One model per author/family
per run. Seed models already in the bestiary are skipped (use `--staples` to
force-include them for regression testing after forge updates). Models that have
permanently failed (forge segfaults, unsupported arch, missing deps) after 2+
attempts are pruned from the seed queue automatically.

**Compilation** — forge backend calls `forge.compile()`; XLA backend JIT-traces
via `jax.jit` on the PJRT TT plugin. Each chip runs its worker as a subprocess;
results are streamed back via a CSV file. A watchdog timer detects when all chips
finish even if a worker subprocess exits silently.

**Backend routing** — in `--backend auto` mode, `router.py` selects forge or xla
per model using a priority chain:

1. JAX/Flax-native models → xla
2. Models with forge crash history (SIGSEGV, forge_internal) → xla
3. Models with XLA runtime error history → forge
4. Architecture XLA affinity (gpt2, bert, albert, etc.) → xla if available
5. Default → forge

---

## Scoring

Points are awarded on compile success. A first compile of a model never before
seen on any Tenstorrent chip is worth more than a thousand words.

| Event | Points |
|---|---|
| Successful compile | +200 base |
| First ever compiled (new to bestiary) | ×5 multiplier → **+1000** |
| Legendary rarity | ×2 |
| Rare | ×1.5 |
| Uncommon | ×1.2 |
| Zero-day model (< 24h old on HF) | +300 |
| Hot (< 1 week) | +100 |
| Fresh (< 1 month) | +50 |
| Consecutive-success streak on same chip | bonus per step |
| First Voice inference produces meaningful output | +100 |

**First Voice** — after a successful compile, each worker runs a themed
inference pass using a curated sample from `lib/expedition/sampler.py`
(stories, images, questions). Decoders in `lib/expedition/decoder.py`
turn raw logits into readable predictions using last-position top-k sampling:
```
🗣 First Voice  [At the Westinghouse pavilion, a time capsule was buried...]
→ The (10%) | A (3%) | " (3%)
```

---

## Side Quests

When a multi-chip model (like BLOOM JAX, which needs all 4 chips simultaneously)
enters the queue, the other chips idle while waiting for the full mesh to
assemble. Instead of wasting that time, Compiletron automatically launches
**side quest** runs — fast, curated image-classification models that keep every
free chip productive until the RALLY quorum is reached.

![Side Quest in action — idle chip picks up a bonus model](docs/demo_side_quest.gif)

**How it works:**

1. A mesh model (e.g. BLOOM JAX requiring 4 chips) is spotted in the queue and
   held as `MESH ASSEMBLING`.
2. Any chip that finishes its main-queue model and would otherwise sit idle is
   instead dispatched a side quest from a curated fast pool.
3. When enough chips free up to form the RALLY quorum, a `_rally_interrupt_flag`
   fires — no new side quests are launched, but in-flight ones run to completion
   before the RALLY begins.
4. Side quest results are tracked separately (`is_sq=True` in RunState) so they
   don't inflate or pollute main-queue metrics. They appear as a compact **⚡ BONUS
   HAUL** line in the Field Report summary.

**Side quest pool** (curated fast models, all single-chip forge):

| Model | Task | Rarity |
|---|---|---|
| MobileNetV2 | image-classification | common |
| GhostNet | image-classification | uncommon |
| GoogLeNet | image-classification | common |
| EfficientNet-Lite | image-classification | uncommon |
| DenseNet-121 | image-classification | uncommon |
| ResNet | image-classification | common |
| SqueezeBERT | text-classification | rare |
| DeiT | image-classification | uncommon |

Side quests are **automatically deduped** — a model already running or completed
on any chip is skipped, so every side quest result is unique within a run.

---

## The Bestiary

`data/bestiary.json` is the persistent record of everything the hardware has
ever learned to compile. Every successful run adds to it; nothing is ever
overwritten. It tracks:

- artifact shape, task, compile time, chip, run number
- first-voice text (the model's first words on Tenstorrent silicon)
- all-time chip leaderboard

Error entries are automatically **re-classified on load** when new error-pattern
rules are added, so stale `other` entries get upgraded to precise categories
over time. The bestiary grows smarter as the project does.

See `data/bestiary.example.json` for the real-world structure with all fields
populated by actual runs.

---

## Backends

### auto (default — intelligent dispatch)

Automatically selects the best backend per model. JAX/Flax models are
routed to xla; PyTorch models go to forge. Forge-fatal and XLA-fatal
history in the bestiary feeds back into routing decisions, so models
that repeatedly crash one backend get redirected to the other. Requires
both backends to be available.

### forge

Uses `tt-forge-fe` to compile PyTorch models via `forge.compile()`.
Requires `source ~/tt-forge-fe/env/activate`.

### xla

Uses `pjrt-plugin-tt` to JIT-compile Flax/JAX models onto TT hardware
via the PJRT plugin interface. Models load via `FlaxAutoModel*` from
transformers. Compiled in a separate `xla-venv` virtualenv.

Three compatibility patches are applied automatically for
pjrt-plugin-tt 0.9.0 + JAX 0.7.1 + Flax 0.8.5:
- `flax.core.tracers.trace_level` — JAX 0.7.x removed `.level` from trace objects
- `jax.local_devices` — redirects cpu-backend requests to tt (only tt is available)
- `_do_init=False` — skips eager Flax init (SliceOp fails in eager mode; JIT works)

### mixed

Even-numbered chips run forge; odd-numbered chips run xla. Useful for
side-by-side comparison of the two compilation stacks.

---

## Recording demos

The TUI has two auto-pilot features that make fully unattended recording possible:

- **Auto-start** — Setup screen counts down 4 seconds then starts the expedition
  automatically. No need to press Enter.
- **`--auto-quit N`** — TUI exits N seconds after the Summary screen appears.
  The recording ends without any manual interaction.

```bash
# Fully unattended: auto-starts, auto-quits 30s after summary (default)
bash scripts/record_demo.sh

# Custom model count
bash scripts/record_demo.sh --models 6

# Custom summary linger time
bash scripts/record_demo.sh --auto-quit 45

# Manual finish (press q on summary screen yourself)
bash scripts/record_demo.sh --no-auto

# Bench mode: 5 timed inference passes per model + stats table at end
bash scripts/record_demo.sh --bench

# Or drive it directly
asciinema rec docs/demo_raw.cast --overwrite \
    --cols 220 --rows 58 \
    --command "python3 expedition.py run --tui \
        --seed-only --limit 16 --chips 4 --no-predownload --auto-quit 30"

# Post-process: smooth and compress
python3 scripts/compress_cast.py docs/demo_raw.cast docs/demo.cast \
    --max-idle 1.2 --min-gap 0.02
```

The `--min-gap` flag floors inter-event gaps to 20 ms, spreading Textual's
async-batched writes into smooth animation rather than single-frame bursts.

---

## Project layout

```
expedition.py               main CLI + TUI launcher
expedition_tui.py           Textual TUI (3 screens)
lib/
  expedition/               expedition subsystems
    bestiary.py             model history database + error re-classification
    decoder.py              logit → text decoder (last-position top-k)
    expedition_worker.py    forge per-chip worker
    expedition_worker_xla.py XLA per-chip worker
    hf_discover.py          HuggingFace frontier scanner
    hud.py                  per-chip run state tracker
    notes.py                run journal writer
    router.py               per-model backend dispatch (auto mode)
    sampler.py              themed inference samples
    scorer.py               rarity/newness scoring
  discovery.py              seed model scanner (tt-forge-models)
  hardware.py               tt-smi hardware detection
data/
  bestiary.json             compiled-model database (gitignored)
  bestiary.example.json     starter bestiary with real-world entries
  perf_history.jsonl        append-only per-run perf timeseries (compile, infer, throughput)
  expeditions/              per-run journals (.md)
  artifacts/                first-voice output archives (.txt)
  runs/                     per-run metadata + example
  samples/                  inference input samples (images, text, audio)
docs/
  demo.cast                 compressed asciinema demo (plays in index.html)
  index.html                demo landing page with asciinema player
scripts/
  compress_cast.py          cast post-processor (max-idle + min-gap smoothing)
  record_demo.sh            one-command demo recorder (no tmux required)
  show_perf_stats.py        display bench stats from perf_history.jsonl
xla-venv/                   separate venv for JAX/PJRT dependencies
requirements.txt            forge-mode dependencies
```

---

## Data files

```
data/bestiary.json          all-time compiled model records + chip scores
data/perf_history.jsonl     append-only per-run performance timeseries
data/expeditions/run_NNN.md per-run journal with first-voice highlights
data/artifacts/             saved first-voice text from notable compiles
```

The bestiary persists across runs and is never overwritten — new compiles
accumulate. It is the canonical record of what the hardware has proven it
can compile.

`perf_history.jsonl` is a separate append-only log: one JSON line per model
per run, written whenever a model compiles successfully. It records
`compile_s`, `infer_s`, `throughput`, `throughput_unit`, and optionally
`bench_passes`, `infer_p50_s`, `infer_p95_s`, `throughput_p50`.

---

## Benchmarking

Add `--bench-passes N` to any run to measure real inference throughput.
After each successful compile, the worker runs 2 warm-up passes then N
timed passes, computing p50/p95 latency and throughput:

```bash
# 5 bench passes per model (2 warm-up + 5 timed)
python3 expedition.py run --tui --bench-passes 5

# With input shape sweep (varies seq_len for LLMs, resolution for vision)
python3 expedition.py run --tui --bench-passes 5 --bench-shapes

# View stats from the last run
python3 scripts/show_perf_stats.py

# View a specific run
python3 scripts/show_perf_stats.py --run 68
```

Throughput unit is automatically determined by model task:
- **tokens/sec** — text-generation, masked-LM, text-classification
- **ms/sample** — image-classification, embeddings, all others

Results are appended to `data/perf_history.jsonl` and rolling-best values
(`best_compile_s`, `best_infer_s`, `best_throughput`) are updated in the
bestiary entry for each model.

---

## Legacy entry point

The original `compiletron.py` / `lib/worker.py` / `lib/models.py` stack
(static 101-model list, tmux 4-pane display) still works but is no longer
the primary interface. `expedition.py` supersedes it with live model
discovery, dynamic queuing, scoring, dual backends, and automatic routing.
