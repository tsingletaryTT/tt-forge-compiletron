# Expedition Mode — Design Spec
**Date:** 2026-04-27  
**Status:** Approved

## Overview

Expedition Mode is a roguelike compilation runner for Tenstorrent hardware. It turns a multi-chip forge compilation run into a scored, persistent game: chips race each other, every new model compiled adds to a growing bestiary, zero-day frontier models yield massive point multipliers, and each run produces a collection of real inference artifacts — text, detections, transcriptions, captions — that persist across sessions.

Classic mode (`compiletron.py`) is untouched. Expedition mode is a new entry point (`expedition.py`) backed by a new `lib/expedition/` subpackage.

---

## Architecture

```
expedition.py               — entry point: hardware detect, queue build, launch tmux
scripts/
  run_expedition.sh         — tmux layout (4-quadrant + shared status strip)
lib/
  expedition/
    __init__.py
    hud.py                  — per-chip score state; writes /tmp/expedition_chip_N.status
    bestiary.py             — persistent bestiary.json + artifact save/load
    decoder.py              — per-task best-effort output decoder
    scorer.py               — points formula, rarity, newness, streak, mesh events
    hf_discover.py          — HF Hub query filtered against bestiary + forge library
data/
  bestiary.json             — persists across all runs
  artifacts/                — one .txt per model (sanitized name key)
  runs/                     — run_NNN.json per run (leaderboard)
```

`expedition.py` orchestrates startup:
1. Detect chips via `lib/hardware.py`
2. Load bestiary from `data/bestiary.json`
3. Build seed queue (tt-forge-models loaders not yet in bestiary)
4. Build frontier queue (HF Hub models not in library or bestiary, sorted by newness × rarity)
5. Interleave seed + frontier, distribute round-robin across chips (mesh-aware)
6. Launch `run_expedition.sh` with per-chip worker scripts

Each chip runs `lib/expedition/expedition_worker.py` — a new script that calls the same `forge.compile()` + inference pipeline as `lib/worker.py` but pipes results through `decoder.py` → `scorer.py` → `hud.py` → writes `/tmp/expedition_chip_N.status`. The shared compile logic lives in a helper importable by both workers to avoid duplication.

---

## Display

### Per-chip pane (one per tmux quadrant)

Model fills the full pane. No per-chip HUD inside the pane — scores live in the shared status strip.

Layout per pane during compilation:
```
[rarity badge]  [model name — large ASCII art or bold text]
[task · param count · source · FIRST EVER / zero-day flag]

[step N/3] [description]
[progress bar]  XX%  XXs elapsed

✓ Architecture loaded: ModelClass
✓ Tokenizer / processor loaded
→ Tracing layer N of M...

last artifact: "[decoded output from previous model]"
```

On success, the pane briefly shows the decoded artifact before moving to the next model. On failure, it shows the error type and moves on — no shame list, no haunting.

### Shared status strip (bottom, full width)

The existing `status_display.sh` pattern extended to show per-chip scores:

```
C0 [████████████░░] 42% ✓9 ✗1  DeepSeek-R1★  pts:2204 🔥×6
C1 [████████░░░░░░] 55% ✓8 ✗0  whisper-v3    pts:1987 🔥×8
C2 [██████░░░░░░░░] 38% ✓7 ✗2  yolov8n       pts:1712 🔥×4  
C3 [████░░░░░░░░░░] 27% ✓6 ✗3  stable-diff   pts:1651 🔥×2
```

During a mesh/multi-chip model, participating chips show `[MESH]` tag and pulse.

### End-of-run summary

Displayed in a new full-screen pane after all chips finish:

```
EXPEDITION #7 COMPLETE  ·  47:32  ·  2026-04-27

🥇 CHIP 0   2,204 pts   ✓12 ✗2   best streak ×6   ★3 first-evers
🥈 CHIP 1   1,987 pts   ✓11 ✗1   best streak ×8   ★2 first-evers
🥉 CHIP 2   1,712 pts   ✓10 ✗3   best streak ×4   ★1 first-ever
   CHIP 3   1,651 pts   ✓9  ✗4   best streak ×3   ★1 first-ever

── NEW TO BESTIARY ────────────────────────────────────────────
★ Qwen2.5-VL-7B:   "The image shows a Tenstorrent chip on a PCB..."
★ whisper-v3:      "Mr. Gorbachev, tear down this wall."
★ yolov8n:         person 0.94, laptop 0.87, cup 0.71, keyboard 0.68
★ DeepSeek-R1:     "To solve this, we first consider the boundary conditions..."
  [... all new models with their artifacts ...]

── FAILED ─────────────────────────────────────────────────────
✗ mistral-7b      RuntimeError: rotary embedding shape mismatch
✗ mochi           TIMEOUT after 90s

── BESTIARY ───────────────────────────────────────────────────
142 / 297 forge-models  ·  31 HF frontier  ·  total 173 compiled
Run #7 personal best: 2,204 pts (Chip 0)
All-time chip leader: Chip 1  (14,203 pts cumulative)
```

---

## Data Model

### `data/bestiary.json`

```json
{
  "compiled": {
    "Qwen/Qwen2.5-VL-7B-Instruct": {
      "first_compiled": "2026-04-27T14:22:00",
      "first_chip": 0,
      "run": 7,
      "best_time_s": 84.2,
      "attempts": 3,
      "successes": 2,
      "source": "huggingface",
      "task": "visual_question_answering",
      "rarity": "rare",
      "hf_downloads": 2400000,
      "hf_created_at": "2025-09-14T00:00:00",
      "artifact": "The image shows a Tenstorrent chip on a PCB..."
    }
  },
  "failed": {
    "mistralai/Mistral-7B-v0.3": {
      "last_error": "RuntimeError: rotary embedding shape mismatch",
      "attempts": 3,
      "run_first_failed": 5
    }
  },
  // failed entries track history for retry interest only — no ongoing penalty, no haunting
  "chip_totals": {
    "0": {"pts": 12204, "first_evers": 18, "best_streak": 9},
    "1": {"pts": 14203, "first_evers": 21, "best_streak": 12},
    "2": {"pts": 9712,  "first_evers": 14, "best_streak": 7},
    "3": {"pts": 8651,  "first_evers": 12, "best_streak": 6}
  }
}
```

### `data/artifacts/<sanitized-model-name>.txt`

Plain text file. First line: metadata header (`model · task · compiled · chip · run`). Remaining lines: decoded artifact content. Epic fails also get an artifact file with the error.

### `data/runs/run_NNN.json`

Per-run record: timestamp, duration, per-chip final scores, new bestiary entries, failures, top artifact. Run number is derived from `len(os.listdir("data/runs")) + 1` at startup and zero-padded to 3 digits.

---

## Scoring

Implemented in `lib/expedition/scorer.py`.

```
base_pts = 50  (any successful compile)

first_ever_bonus = 100  (model not in bestiary)

rarity_multiplier:
  legendary  (>10M HF downloads)  ×4
  rare       (1M–10M)             ×2
  uncommon   (100K–1M)            ×1.5
  common     (<100K)              ×1.0
  familiar   (tt-forge-models, not on HF)  ×1.0

newness_multiplier (HF model age — applies to first-ever compiles only; repeat compiles always ×1.0):
  zero-day   (<24h old)    ×5  — banner celebration, special ASCII art
  hot        (<7 days)     ×3
  fresh      (<30 days)    ×2
  recent     (<90 days)    ×1.5
  established (90d+)       ×1.0
  familiar   (forge-models, no HF date)  ×1.0

streak_multiplier:
  N consecutive successes: min(1.0 + N×0.1, 2.0)

mesh_event_bonus:
  2-chip model:   each participating chip gets full points
  4-chip / T3K:   each chip gets full points + 50pt mesh bonus
  Galaxy / large: full points + 200pt event bonus

failure: −10 pts (no additional penalty)

final_pts = (base_pts + first_ever_bonus) × rarity_mult × newness_mult × streak_mult
            + mesh_event_bonus (if applicable)
```

Zero-day example: `(50 + 100) × 4 × 5 × 1.5 = 4,500 pts`

---

## HF Discovery

Implemented in `lib/expedition/hf_discover.py`.

At run start, query `HfApi.list_models(filter="pytorch", sort="createdAt", direction=-1, limit=500)`. This returns newest models first — maximizing zero-day and hot finds.

Filtering:
1. Cross-reference against bestiary → skip already-compiled
2. Cross-reference against tt-forge-models library → these become seed queue instead
3. Remaining → frontier queue, sorted by `createdAt` descending (newest first)

Dynamic loader construction for frontier models:
- Read `pipeline_tag` from model card → map to `AutoModel` class and dummy input shape
- Supported pipeline tags: `text-generation`, `fill-mask`, `question-answering`, `image-classification`, `object-detection`, `automatic-speech-recognition`, `image-to-text`, `text-to-image`, `depth-estimation`, `image-segmentation`
- Unsupported tags: skip, log as `skipped_unsupported_task`
- Multi-chip hint detection: flag models with `>40B params` or known MoE architectures for mesh assignment

Queue interleaving: distribute 60% seed, 40% frontier to each chip. Frontier models go highest-rarity/newest first.

---

## Output Decoder

Implemented in `lib/expedition/decoder.py`.

Signature: `decode(output_tensors, model_info: ModelInfo | FrontierModelInfo, inputs) -> str`

`FrontierModelInfo` is a lightweight dataclass constructed at runtime for HF frontier models: `name`, `task` (from `pipeline_tag`), `source="huggingface"`. It implements the same `task` attribute interface as `ModelInfo` so the decoder dispatch works identically for both.

| ModelTask | Decode strategy |
|---|---|
| `CAUSAL_LM`, `SEQ2SEQ_LM`, `CAUSAL_LM_WITH_PAST` | Greedy decode first 100 chars via tokenizer |
| `MASKED_LM` | Top predicted token at `[MASK]` |
| `QUESTION_ANSWERING` | Extract answer span from start/end logits |
| `IMAGE_CLASSIFICATION` | Top-3 class labels + confidence |
| `OBJECT_DETECTION` | Objects with confidence ≥ 0.5 (label + score) |
| `SEMANTIC_SEGMENTATION`, `PANOPTIC_SEGMENTATION` | Unique class labels in output mask |
| `DEPTH_ESTIMATION` | Min/max depth range |
| `AUTOMATIC_SPEECH_RECOGNITION` | Decoded transcription |
| `AUDIO_CLASSIFICATION` | Top audio class |
| `TEXT_TO_SPEECH` | Duration + sample rate |
| `IMAGE_TO_TEXT`, `VISUAL_QA`, `IMAGE_CAPTIONING` | First 100 chars of generated text |
| `IMAGE_GENERATION` | Tensor shape + pixel value range |
| Any / error | `shape={shape} dtype={dtype} range=[{min:.2f}, {max:.2f}]` |

All decode errors caught silently → raw fallback. The artifact string is saved to `data/artifacts/` and shown in the run summary.

---

## Mesh-Aware Distribution

`expedition.py` checks `model.get_mesh_config()` before distributing:

- **Single-chip**: assigned to one chip, standard scoring
- **2-chip**: claims two adjacent chips; both panes show the model; points awarded to both chips
- **4-chip / T3K**: all chips participate; shown as a team event; each chip earns full points + mesh bonus
- **Galaxy / large mesh**: special event banner; each chip earns full points + large mesh bonus
- **HF frontier models**: default single-chip unless param count or architecture hints at tensor parallelism

During a mesh model, the status strip shows `[MESH 2×]`, `[MESH 4×]`, etc. on participating chips.

---

## Rarity Reveal

When a first-ever model starts compiling, the pane gets a brief rarity reveal before the progress bar:

- **Common / Familiar**: standard banner, no fanfare
- **Uncommon**: yellow `◆ UNCOMMON` badge
- **Rare**: pink `★ RARE FIND` badge, 1s pause
- **Legendary**: full-width ASCII art celebration, 2s pause, teal color scheme
- **Zero-day**: full-width `⚡ ZERO DAY` banner, 3s pause, golden color scheme

---

## Files Changed / Created

**New files:**
- `expedition.py`
- `lib/expedition/__init__.py`
- `lib/expedition/expedition_worker.py`
- `lib/expedition/hud.py`
- `lib/expedition/bestiary.py`
- `lib/expedition/decoder.py`
- `lib/expedition/scorer.py`
- `lib/expedition/hf_discover.py`
- `scripts/run_expedition.sh`
- `data/bestiary.json` (created on first run)
- `data/runs/` (directory, created on first run)
- `data/artifacts/` (directory, created on first run)

**Unchanged:**
- `compiletron.py`
- `lib/worker.py`
- `lib/models.py`
- `lib/hardware.py`
- `lib/discovery.py`
- `scripts/run_4way_tmux.sh`

**Possibly extended:**
- `scripts/status_display.sh` — add score/streak columns to existing chip status bars
