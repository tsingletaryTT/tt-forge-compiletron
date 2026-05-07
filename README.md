# TT-Forge Compiletron

A competitive model-compilation game for Tenstorrent hardware. Discovers models
from HuggingFace and the tt-forge-models zoo, compiles them across all available
chips in parallel, scores results by rarity and novelty, and maintains a bestiary
of everything that has ever compiled.

Supports two compilation backends — **tt-forge** (PyTorch via forge) and **tt-xla**
(JAX/Flax via PJRT plugin) — selectable per-chip or in mixed mode.

<img width="3840" height="2002" alt="tt-forge-compiletron TUI" src="https://github.com/user-attachments/assets/3e93d7d6-8e02-49f6-92cb-e2d93c6caec2" />

**Tested on:** 4× P300C Blackhole chips

---

## Quick start

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron
pip install -r requirements.txt

# Activate tt-forge backend
source ~/tt-forge-fe/env/activate

# Launch TUI (recommended)
python3 expedition.py run --tui

# Or CLI, 4-chip forge run, 20 models
python3 expedition.py run --chips 4 --limit 20
```

### XLA backend (JAX/PJRT)

```bash
# One-time setup
python3 -m venv xla-venv
xla-venv/bin/pip install pjrt-plugin-tt jax==0.7.1 jaxlib==0.7.1 \
    flax==0.8.5 "transformers<5.0" torch --index-url https://pypi.tenstorrent.com/simple/

# Run with XLA backend
python3 expedition.py run --tui --backend xla
```

---

## The TUI

`expedition.py run --tui` opens a 3-screen Textual app:

```
╔══════════════════════════════════════════
║  EXPEDITION #007 SETUP

  Seeds: tt-forge-models + HuggingFace frontier
  Backend: forge  [cycle with 5]
  Chips:   4      Limit: 20

  [Enter] Start   [Q] Quit
╚══════════════════════════════════════════
```

**Setup screen** — configure chips, limit, backend (forge / xla / mixed),
and source filters. Press Enter to start.

**Run screen** — one panel per chip, live event log, scrolling compilation
banners, real-time scores, and First Voice inference output.

**Summary screen** — points leaderboard by chip, compile-time histogram,
failure details, all-time bestiary stats.

---

## CLI reference

```bash
# Run modes
python3 expedition.py run --tui                    # interactive TUI
python3 expedition.py run --chips 4 --limit 20     # CLI, 4 chips, 20 models
python3 expedition.py run --backend xla            # JAX/PJRT backend
python3 expedition.py run --backend mixed          # even chips=forge, odd=xla
python3 expedition.py run --seed-only              # tt-forge-models zoo only
python3 expedition.py run --frontier-only          # HuggingFace frontier only
python3 expedition.py run --staples                # re-run proven seed models

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
force-include them).

**Compilation** — forge backend calls `forge.compile()`; XLA backend JIT-traces
via `jax.jit` on the PJRT TT plugin. Each chip runs its worker as a subprocess;
results are streamed back via a CSV file.

**Scoring** — points are awarded on compile success:
- Base: +200 pts
- First-ever compiled: ×5 bonus (1000 pts)
- Rarity tiers: legendary (×2), rare (×1.5), uncommon (×1.2)
- Newness: zero-day (+300), hot (+100), fresh (+50)
- Streak: 🔥 bonus for consecutive successes on same chip
- First Voice: +100 pts if inference produces meaningful output

**Bestiary** (`data/bestiary.json`) — persistent database. Tracks every
model ever compiled: artifact shape, task, compile time, chip, run number,
first-voice text, and all-time chip leaderboard.

**First Voice** — after a successful compile, each worker runs a themed
inference pass using a curated sample from `lib/expedition/sampler.py`
(stories, images, questions). Decoders in `lib/expedition/decoder.py`
turn raw logits into readable predictions, e.g.:
```
🗣 First Voice  [At the Westinghouse pavilion, a time capsule was buried...]
→ The (10%) | A (3%) | " (3%)
```

---

## Backends

### auto (default — intelligent dispatch)

Automatically selects the best backend per model. JAX/Flax models are
routed to xla; PyTorch models go to forge. Falls back to forge when
affinity is ambiguous. Requires both backends to be available.

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

## Project layout

```
expedition.py               main CLI + TUI launcher
expedition_tui.py           Textual TUI (3 screens)
lib/
  expedition/               expedition subsystems
    bestiary.py             model history database
    decoder.py              logit → text decoder (First Voice)
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
  bestiary.json             compiled-model database
  expeditions/              per-run journals
  artifacts/                first-voice output archives
  runs/                     per-chip result CSVs
xla-venv/                   separate venv for JAX/PJRT dependencies
requirements.txt            forge-mode dependencies
```

---

## Data files

```
data/bestiary.json          all-time compiled model records + chip scores
data/expeditions/run_NNN.md per-run journal with first-voice highlights
```

The bestiary persists across runs and is never overwritten — new compiles
accumulate. It is the canonical record of what the hardware has proven it
can compile.

---

## Legacy entry point

The original `compiletron.py` / `lib/worker.py` / `lib/models.py` stack
(static 101-model list, tmux 4-pane display) still works but is no longer
the primary interface. `expedition.py` supersedes it with live model
discovery, dynamic queuing, scoring, and dual backends.
