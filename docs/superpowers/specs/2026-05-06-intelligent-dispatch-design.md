# Intelligent Dispatch + Multi-Chip Mesh + XLA First-Class Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make backend selection (forge-onnx vs tt-xla) and chip count fully automatic per model, implement real multi-chip mesh dispatch with "RALLY" game mechanics, and close the three XLA first-class infrastructure gaps.

**Architecture:** A new `router.py` module computes a `DispatchDecision` per model before it enters the pool. The TUI becomes a real-time dispatcher (per-model subprocesses instead of per-chip full-queue processes). Mesh coordination lives in the TUI's async event loop — when quorum is met, a single multi-chip subprocess fires with a full-width RALLY banner replacing the chip grid. XLA infrastructure gaps (bestiary backend field, Flax frontier discovery, pre-download patterns) are fixed as part of the same pass.

**Tech Stack:** Python 3.11, Textual 0.x, asyncio, JAX 0.7.1/pjrt-plugin-tt 0.9.0, tt-forge-fe, huggingface_hub

---

## Files Changed or Created

| File | Change |
|---|---|
| `lib/expedition/router.py` | **NEW** — `DispatchDecision` + `route_model()` |
| `lib/expedition/scorer.py` | `mesh_mult` multiplier, `opportunist_bonus`, `formation_share` |
| `lib/expedition/bestiary.py` | `backend` param on `record_success()` |
| `lib/expedition/hf_discover.py` | `library` param on `discover_frontier()` + `discover_from_authors()` |
| `expedition.py` | Split `_IGNORE_PATTERNS`, thread `library` through `_scan_frontier()` |
| `expedition_tui.py` | Per-model dispatch loop, RALLY banner, SetupScreen auto mode, confidence labels |
| `lib/expedition/expedition_worker.py` | Accept `--model-json` single-model mode |
| `lib/expedition/expedition_worker_xla.py` | Accept `--model-json` single-model mode |

---

## Section 1 — `lib/expedition/router.py` (new file)

Single responsibility: given a queue item dict and a Bestiary, return a routing decision. Nothing else imports routing logic; this module imports nothing from the project (only stdlib + bestiary type).

### `DispatchDecision` dataclass

```python
@dataclass
class DispatchDecision:
    backend: str        # "forge" or "xla"
    chips: int          # 1, 2, or 4
    confidence: float   # 0.0–1.0, informational only
    reason: str         # short label for UI: "jax-native", "params-40b", "forge-failure-history", etc.
```

### `route_model(item: dict, bestiary: Bestiary) -> DispatchDecision`

Applies signals in priority order — first match wins:

**Backend signals (highest priority first):**

1. `item["library"] == "jax"` or `item["library"] == "flax"` → backend=`"xla"`, confidence=0.92, reason=`"jax-native"`
2. Bestiary failure history: `bestiary.failed[model_id]` has ≥2 failures with `error_category` in `{"forge_missing_op", "forge_internal"}` → backend=`"xla"`, confidence=0.75, reason=`"forge-failure-history"`
3. `item.get("model_type")` in `_XLA_AFFINITY_TYPES` (small table, see below) → backend=`"xla"`, confidence=0.68, reason=`"arch-xla-affinity"`
4. Default → backend=`"forge"`, confidence=0.60, reason=`"default"`

**`_XLA_AFFINITY_TYPES`:** `{"flax_bert", "flax_gpt2", "flax_roberta", "flax_t5"}` — Flax-canonical architectures that have already proven out on tt-xla.

**Chip count signals:**

1. `item.get("mesh_chips", 1)` already computed by `hf_discover.py` (MoE name heuristic + params_b > 40B → 4 chips) — adopt directly
2. `params_b` fallback: 7–40B → 2 chips; else keep mesh_chips value
3. If `chips > len(available_chips)` → cap at `len(available_chips)`

`available_chips` is passed in from the TUI at route time (the set of all chip IDs in this run, not just free ones).

---

## Section 2 — Scoring changes (`lib/expedition/scorer.py`)

### Mesh multiplier replaces flat mesh bonus

Old formula:
```
pts = int((base + first_ever + first_voice) × rarity × newness × streak) + mesh_bonus
```

New formula:
```
pts = int((base + first_ever + first_voice) × rarity × newness × streak × mesh_mult)
```

Where `mesh_mult = 1.0 + (chips − 1) × 0.5`:
- 1 chip → 1.0× (no change from current single-chip scores)
- 2 chips → 1.5×
- 4 chips → 2.5×

The old `mesh_bonus` dict (`{≥4: 50, ≥32: 200}`) is removed.

### New scoring parameters

`compute_score()` gains two new bool params:

```python
def compute_score(
    success: bool,
    is_first_ever: bool,
    rarity: Rarity,
    newness: Newness,
    streak: int,
    mesh_chips: int = 1,
    is_first_voice: bool = False,
    is_opportunist: bool = False,   # NEW: compiled while mesh was assembling
    is_formation_share: bool = False,  # NEW: non-lead chip in a mesh compile
) -> ScoreResult:
```

- `is_opportunist=True` → add flat +25 to pts (added after the multiplier bracket, as an integer bonus)
- `is_formation_share=True` → pts = 150 flat (replaces the normal formula entirely — these chips contributed muscle, not strategy)

`ScoreResult.breakdown` gains `"opportunist_bonus"` and `"formation_share"` keys.

---

## Section 3 — Dispatch architecture (`expedition_tui.py`)

### Workers become single-model subprocesses

Both workers gain a `--model-json <path>` argument. When present, they load that single JSON file as the queue (a list with one item) instead of `--queue`. The existing queue-loop in each worker runs unchanged — it just processes one item and exits.

`--queue` remains supported for backwards compatibility with the CLI path (`expedition.py run` without TUI).

### TUI dispatcher state

`RunScreen` gains:

```python
self._model_pool: list[dict]        # all pending models, not yet dispatched
self._free_chips: set[int]          # chip IDs with no running worker
self._mesh_holding: dict | None     # mesh model waiting for quorum
self._opportunist_active: bool      # True while _mesh_holding is set
self._chip_procs: dict[int, asyncio.subprocess.Process]  # running procs
```

### `on_mount` change

Old: `for chip_id in range(self.num_chips): self._launch_chip(chip_id)`

New:
1. Build `self._model_pool` from `chip_queues` (flatten the round-robin into one ordered list)
2. Mark all chips as free: `self._free_chips = set(range(self.num_chips))`
3. Call `self._dispatch_next()` once to seed initial work

### `_dispatch_next()`

Called whenever a chip completes or on initial mount. Logic:

```python
def _dispatch_next(self):
    # 1. Check if mesh quorum is met
    if self._mesh_holding:
        chips_needed: int = self._mesh_holding["chips_needed"]  # int, e.g. 4
        if len(self._free_chips) >= chips_needed:
            chip_ids = sorted(self._free_chips)[:chips_needed]
            self._fire_rally(self._mesh_holding, chip_ids)
            return

    # 2. Find next dispatchable model from pool
    for i, model in enumerate(self._model_pool):
        decision = route_model(model, self._bestiary)
        chips_needed = set(range(decision.chips))  # uses lowest available chip IDs

        if decision.chips == 1:
            if self._free_chips:
                chip_id = min(self._free_chips)
                self._model_pool.pop(i)
                self._launch_model(chip_id, model, decision)
                return
        else:
            # Multi-chip model: hold it, keep dispatching singles past it
            if self._mesh_holding is None:
                self._mesh_holding = {**model, "chips_needed": decision.chips, "decision": decision}
                self._opportunist_active = True
                el = self.query_one("#event-log", EventLog)
                el.write(f"[yellow]⏳ MESH ASSEMBLING — {model['model_id'].split('/')[-1]} needs {decision.chips} chips[/]")
            continue  # keep scanning pool for single-chip models

    # 3. Nothing dispatchable — chip goes idle
```

### `_launch_model(chip_id, model, decision, mesh_chip_ids=None)`

Replaces the old `_launch_chip`. Writes `model` to `/tmp/expedition_model_chip{chip_id}.json`, launches the appropriate worker subprocess with `--model-json`, streams output to the chip's panel.

When `mesh_chip_ids` is provided (a list of ints, e.g. `[0,1,2,3]`):
- Sets `TT_VISIBLE_DEVICES=",".join(str(c) for c in mesh_chip_ids)` (e.g. `"0,1,2,3"`)
- Streams output to the lead chip panel (`#chip-{chip_id}`)
- On process exit: scores the compile with `mesh_chips=len(mesh_chip_ids)` for the lead chip (full `mesh_mult`), then for each non-lead chip in `mesh_chip_ids` calls `bestiary.add_chip_points(chip=cid, pts=150, ...)` with `is_formation_share=True`; then calls `_on_chip_free(cid)` for all mesh chips

**Initial stagger:** the first dispatch to each chip (when `_model_pool` is being seeded at mount) adds `await asyncio.sleep(chip_id * 2)` before launching. Subsequent dispatches to that chip have no delay.

On process exit (single-chip) → calls `_on_chip_free(chip_id)`.

### `_on_chip_free(chip_id)`

```python
def _on_chip_free(self, chip_id: int):
    self._free_chips.add(chip_id)
    self._done_count += 1
    self._dispatch_next()
```

### `_fire_rally(mesh_model, chip_ids)`

```python
def _fire_rally(self, model: dict, chip_ids: list[int]):
    self._mesh_holding = None
    self._opportunist_active = False
    for cid in chip_ids:
        self._free_chips.discard(cid)

    # Show RALLY banner (replaces chip grid)
    self.query_one("#chip-grid").display = False
    self.query_one("#rally-banner").display = True
    rally = self.query_one("#rally-banner", RallyBanner)
    rally.start(model, chip_ids, decision)

    # Launch single multi-chip subprocess
    # chip 0 of chip_ids is the lead — its panel gets live output
    lead = chip_ids[0]
    chip_ids_str = ",".join(str(c) for c in chip_ids)
    self._launch_model(lead, model, decision, mesh_chip_ids=chip_ids)
    # _launch_model sets TT_VISIBLE_DEVICES=chip_ids_str when mesh_chip_ids is provided
```

### `RallyBanner` widget (new)

A `Static` widget (initially `display = False`) that covers the full width of `#chip-grid` when a RALLY fires. Contains:

```
╔══════════════════════════════════════════════════════════
║  ⚡⚡ RALLY — CHIPS 0·1·2·3 ASSEMBLED ⚡⚡
║  deepseek-v3  ·  4-chip mesh  ·  forge  ·  conf 0.74
║
║  ▶ [forge] Compiling on mesh 0,1,2,3...
║  [live output streams here]
╚══════════════════════════════════════════════════════════

  CHIP 0 ████  CHIP 1 ████  CHIP 2 ████  CHIP 3 ████
  [dim panels below show locked state]
```

On mesh worker exit: hide `#rally-banner`, show `#chip-grid`, call `_on_chip_free(cid)` for all mesh chip IDs, score the compile with `mesh_mult` and lead chip gets full score, all non-lead chips get `formation_share=True` score (+150 flat).

### SetupScreen backend cycle

Old: `forge → xla → mixed → forge`
New: `auto → forge → xla → mixed → auto`

`auto` is the new default (`self._backend = "auto"`). Display label: `Backend: [bold]AUTO[/]  [dim]routes per-model[/]`.

In `_do_setup_body`, when `self._backend == "auto"`, pass `backend="auto"` to `RunScreen`.

In `RunScreen._dispatch_next()` (and `_launch_model`), the routing branch is:
```python
if self.backend == "auto":
    decision = route_model(model, self._bestiary, available_chips=set(range(self.num_chips)))
else:
    # Manual override: honour user's backend choice, use mesh_chips from model metadata
    decision = DispatchDecision(
        backend=self.backend,
        chips=model.get("mesh_chips", 1),
        confidence=1.0,
        reason="manual",
    )
```

`RunScreen` loads the bestiary at mount for router queries: `self._bestiary = Bestiary(path=str(self._project_dir / "data" / "bestiary.json"))`.

### Confidence label in chip panels

When `_launch_model` starts, it writes one dim line to the chip panel:
```
  [dim]routing: forge · conf 0.82 · 1-chip[/]
```

---

## Section 4 — Bestiary backend field (`lib/expedition/bestiary.py`)

### `record_success()` signature change

Add `backend: str = "forge"` parameter. Store it on the compiled entry:

```python
self._data["compiled"][model_id] = {
    ...existing fields...,
    "backend": backend,   # NEW
}
```

On subsequent calls (model already in compiled), do NOT overwrite `backend` — it records the first successful backend. Add a `backends_succeeded: list[str]` field that accumulates all backends that have compiled this model:

```python
entry.setdefault("backends_succeeded", [entry.get("backend", "forge")])
if backend not in entry["backends_succeeded"]:
    entry["backends_succeeded"].append(backend)
```

Both workers pass `backend=BACKEND_LABEL` (forge worker uses `"forge"`, xla worker uses `"xla"`).

### `is_compiled_by(model_id, backend)` helper

```python
def is_compiled_by(self, model_id: str, backend: str) -> bool:
    entry = self._data["compiled"].get(model_id)
    if not entry:
        return False
    return backend in entry.get("backends_succeeded", [entry.get("backend", "forge")])
```

Used by the router to check if forge already failed AND xla already succeeded (skip re-queuing).

---

## Section 5 — XLA frontier discovery (`lib/expedition/hf_discover.py`)

### `discover_frontier()` signature change

Add `library: str | None = "pytorch"` parameter. Pass it to the HF API call:

```python
hf_models = api.list_models(
    filter=library,   # was hardcoded "pytorch"
    ...
)
```

When `library=None` (mixed mode), omit the filter entirely.

### `discover_from_authors()` signature change

Same change: `library: str | None = "pytorch"` → pass to `api.list_models(filter=library, ...)`.

### Thread through `_scan_frontier()` (`expedition.py`)

```python
def _scan_frontier(
    bestiary_compiled_ids, forge_model_ids,
    ...,
    library: str | None = "pytorch",   # NEW
) -> list[dict]:
    models = discover_frontier(..., library=library)
    ...supplement = discover_from_authors(..., library=library)
```

TUI's `_scan_frontier()` call passes `library=scan_fw` where `scan_fw` is already computed as `{"forge": "pytorch", "xla": "jax", "mixed": None, "auto": None}[self._backend]`.

---

## Section 6 — Pre-download pattern split (`expedition.py`)

Replace single `_IGNORE_PATTERNS` list with two backend-specific lists:

```python
# Patterns to ignore when pre-downloading for forge-onnx backend.
# Forge only needs PyTorch safetensors — skip Flax/TF/Keras formats.
_FORGE_IGNORE_PATTERNS = [
    "*.msgpack",    # Flax/JAX checkpoints (not needed by forge)
    "flax_model*",  # Flax model shards
    "*.h5",         # Keras/TF HDF5 weights
    "tf_model*",    # TensorFlow SavedModel
    "rust_model*",  # Rust/candle weights
    "*.ot",         # OpenNMT tokenizer files
]

# Patterns to ignore when pre-downloading for tt-xla backend.
# XLA needs Flax weights (.msgpack) — skip TF/Keras/Rust/PyTorch-only formats.
_XLA_IGNORE_PATTERNS = [
    "*.h5",         # Keras/TF HDF5 weights
    "tf_model*",    # TensorFlow SavedModel
    "rust_model*",  # Rust/candle weights
    "*.ot",         # OpenNMT tokenizer files
]
```

The pre-download call passes `ignore_patterns=_XLA_IGNORE_PATTERNS if backend == "xla" else _FORGE_IGNORE_PATTERNS`.

In `auto` mode, use `_FORGE_IGNORE_PATTERNS` for models routed to forge, `_XLA_IGNORE_PATTERNS` for models routed to xla. Since routing happens before pre-download in the TUI's setup flow, each model's `DispatchDecision` is available at download time.

---

## Section 7 — Worker single-model mode

Both `expedition_worker.py` and `expedition_worker_xla.py` gain:

```python
parser.add_argument("--model-json", default=None,
    help="Path to a JSON file with a single model dict. Overrides --queue.")
```

In the worker's queue-loading section:

```python
if args.model_json:
    queue = [json.loads(Path(args.model_json).read_text())]
else:
    queue = json.loads(Path(args.queue).read_text())
```

The rest of the worker loop is unchanged — it processes each item in `queue`, which in single-model mode has exactly one entry.

---

## Scoring example: legendary 4-chip RALLY

Model: `deepseek-v3`, legendary (≥10M downloads), zero-day, first-ever, first-voice, 4-chip mesh, streak=3.

```
base              =   50
first_ever_bonus  =  100
first_voice_bonus =  100
  subtotal        =  250

rarity_mult       = 4.0   (legendary)
newness_mult      = 5.0   (zero-day, first-ever)
streak_mult       = 1.3   (streak=3, capped at 2.0)
mesh_mult         = 2.5   (4 chips)

pts = int(250 × 4.0 × 5.0 × 1.3 × 2.5) = int(16,250) = 16,250  (lead chip)

formation_share chips (1, 2, 3): +150 each
```

For comparison, the same model as a single-chip compile: `int(250 × 4.0 × 5.0 × 1.3 × 1.0)` = 6,500 pts.

---

## Verification

```bash
# Syntax checks
python3 -c "
import py_compile
for f in [
    'lib/expedition/router.py',
    'lib/expedition/scorer.py',
    'lib/expedition/bestiary.py',
    'lib/expedition/hf_discover.py',
    'expedition.py',
    'expedition_tui.py',
    'lib/expedition/expedition_worker.py',
    'lib/expedition/expedition_worker_xla.py',
]:
    py_compile.compile(f, doraise=True)
    print('ok', f)
"

# Router unit test
python3 -c "
from lib.expedition.router import route_model, DispatchDecision
from lib.expedition.bestiary import Bestiary
b = Bestiary()
# jax-native model
d = route_model({'model_id': 'google/flax-bert', 'library': 'jax', 'hf_downloads': 5000, 'mesh_chips': 1}, b)
assert d.backend == 'xla', d
assert d.chips == 1
print('router: jax-native → xla OK')
# large params model
d = route_model({'model_id': 'deepseek-ai/deepseek-v3', 'library': 'pytorch', 'hf_downloads': 1e6, 'mesh_chips': 4, 'hf_params_b': 67.0}, b)
assert d.chips == 4, d
print('router: large params → 4 chips OK')
# default
d = route_model({'model_id': 'foo/bar', 'library': 'pytorch', 'hf_downloads': 100, 'mesh_chips': 1}, b)
assert d.backend == 'forge', d
print('router: default → forge OK')
"

# Scorer mesh_mult test
python3 -c "
from lib.expedition.scorer import compute_score, Rarity, Newness
s1 = compute_score(True, True, Rarity.COMMON, Newness.FRESH, 0, mesh_chips=1)
s4 = compute_score(True, True, Rarity.COMMON, Newness.FRESH, 0, mesh_chips=4)
assert s4.pts == int(s1.pts * 2.5), f'{s4.pts} != {int(s1.pts * 2.5)}'
print(f'scorer: mesh_mult OK  1-chip={s1.pts}  4-chip={s4.pts}')
opp = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, is_opportunist=True)
assert opp.breakdown['opportunist_bonus'] == 25
print('scorer: opportunist bonus OK')
share = compute_score(True, False, Rarity.COMMON, Newness.ESTABLISHED, 0, is_formation_share=True)
assert share.pts == 150
print('scorer: formation share OK')
"

# TUI import
python3 -c "from expedition_tui import ExpeditionTUI; print('TUI import OK')"

# Dry run (no hardware needed)
python3 expedition.py run --seed-only --limit 3 --chips 1 --no-predownload
```
