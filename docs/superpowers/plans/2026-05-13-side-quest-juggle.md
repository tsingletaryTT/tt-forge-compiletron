# "Juggle While We Wait" — Side Quest Speed Run

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When chips go idle waiting for a multi-chip RALLY (e.g. BLOOM 4-chip), automatically start a "HuggingFace side quest" speed run on idle chips — compiling tiny popular models for bonus points, then stopping cleanly when all chips are called to RALLY.

**Architecture:** Side quests launch in the same `_launch_model` subprocess infrastructure but tracked separately (`_side_quest_chips`, `_side_quest_procs`). The RALLY interrupt flag stops side quest workers cleanly. The dispatcher (`_dispatch_next`) includes side quest chips in the RALLY quorum check so the RALLY fires as soon as all main-queue work is done regardless of side quest state.

**Tech Stack:** Python asyncio, Textual `@work`, existing forge worker subprocess, existing RunScreen dispatch loop.

---

## File map

| File | Changes |
|---|---|
| `expedition.py` | Add `_build_side_quest_pool()`, change `_build_curated_queue()` to return `(queues, pool)` tuple, update caller |
| `expedition_tui.py` | `SetupScreen._do_setup_body`: unpack tuple, pass pool to `_advance_to_run`. `_advance_to_run`: accept + pass pool. `RunScreen.__init__`: accept `side_quest_pool`. `RunScreen._dispatch_next`: inject side quests + include in RALLY quorum. `RunScreen._fire_rally`: kill side quest procs. New `RunScreen._launch_side_quest` `@work` method |

---

## Task 1 — Side quest model pool (`expedition.py`)

**Files:**
- Modify: `expedition.py` (add `_build_side_quest_pool`, update `_build_curated_queue`, update caller at ~line 1601)

### What the pool looks like
Each entry follows the same dict schema as curated queue items. All are `mesh_chips=1`, `library="pytorch"`, forge worker, `source="tt-forge-models"`. These models load cleanly (verified by import test):

```python
_FORGEMS = "_forgems"
pool = [
    {
        "model_id": "mobilenetv2/pytorch",
        "display_name": "MobileNetV2",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "common",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.mobilenetv2.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "ghostnet/pytorch",
        "display_name": "GhostNet",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "uncommon",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.ghostnet.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "googlenet/pytorch",
        "display_name": "GoogLeNet",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "common",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.googlenet.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "efficientnet_lite/pytorch",
        "display_name": "EfficientNet-Lite",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "uncommon",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.efficientnet_lite.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "densenet/pytorch",
        "display_name": "DenseNet-121",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "uncommon",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.densenet.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "resnet/pytorch",
        "display_name": "ResNet",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "common",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.resnet.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "squeezebert/pytorch",
        "display_name": "SqueezeBERT",
        "task": "text-classification",
        "source": "tt-forge-models",
        "rarity": "rare",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.squeezebert.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
    {
        "model_id": "deit/pytorch",
        "display_name": "DeiT",
        "task": "image-classification",
        "source": "tt-forge-models",
        "rarity": "uncommon",
        "hf_downloads": None,
        "hf_created_at": None,
        "mesh_chips": 1,
        "library": "pytorch",
        "model_type": "",
        "loader_module": f"{_FORGEMS}.deit.pytorch.loader",
        "loader_class": "ModelLoader",
        "is_frontier": False,
    },
]
```

- [ ] **Step 1: Add `_build_side_quest_pool(num_chips: int) -> list[dict]` after `_build_curated_queue`**

Add this function at `expedition.py` immediately after `_build_curated_queue` (currently ~line 588):

```python
def _build_side_quest_pool(num_chips: int) -> list[dict]:
    """Return curated tiny models for idle-chip side-quest speed runs.

    These are small, fast-compiling forge models used when chips become idle
    while waiting for a multi-chip RALLY.  All use mesh_chips=1 and the
    standard forge (PyTorch) worker.
    """
    _FORGEMS = "_forgems"
    return [
        # -- paste the full list from Task 1 above --
    ]
```

(Use the full pool dict from above — all 8 entries.)

- [ ] **Step 2: Change `_build_curated_queue` return from `list[list[dict]]` to `tuple[list[list[dict]], list[dict]]`**

At the end of `_build_curated_queue`, replace:
```python
    return chip_queues
```
with:
```python
    return chip_queues, _build_side_quest_pool(num_chips)
```

Update the docstring's return description accordingly.

- [ ] **Step 3: Update the non-TUI caller in `expedition.py` (~line 1601)**

```python
# Before:
chip_queues = _build_curated_queue(num_chips)
# After:
chip_queues, _ = _build_curated_queue(num_chips)
```
(Non-TUI scrolling mode doesn't use side quests — pool is discarded.)

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('expedition.py', doraise=True); print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Verify pool builds correctly**

```bash
python3 -c "
from expedition import _build_curated_queue, _build_side_quest_pool
queues, pool = _build_curated_queue(4)
print(f'queues: {[len(q) for q in queues]}')
print(f'pool: {len(pool)} models')
for m in pool:
    print(f'  {m[\"display_name\"]:25s} {m[\"model_id\"]}')
"
```

Expected: `queues: [2, 1, 1, 1]`, `pool: 8 models`, each model listed.

- [ ] **Step 6: Commit**

```bash
git add expedition.py
git commit -m "Add _build_side_quest_pool: 8 curated fast models for idle-chip juggle"
```

---

## Task 2 — Thread side quest pool from SetupScreen to RunScreen (`expedition_tui.py`)

**Files:**
- Modify: `expedition_tui.py` (~lines 665-833, SetupScreen._do_setup_body, _advance_to_run, RunScreen.__init__)

The pool produced in Task 1 needs to flow: `_build_curated_queue()` → `_do_setup_body` → `_advance_to_run` → `RunScreen.__init__`.

- [ ] **Step 1: Update `_do_setup_body` curated branch (~line 667)**

```python
# Before:
chip_queues = _build_curated_queue(self._chips)
...
app.call_from_thread(self._advance_to_run, chip_queues)

# After:
chip_queues, side_quest_pool = _build_curated_queue(self._chips)
...
app.call_from_thread(self._advance_to_run, chip_queues, side_quest_pool)
```

- [ ] **Step 2: Update `_advance_to_run` signature and RunScreen constructor call (~line 815)**

```python
# Before:
def _advance_to_run(self, chip_queues: list[list[dict]]) -> None:
    ...
    self.app.push_screen(
        RunScreen(
            chip_queues  = chip_queues,
            num_chips    = self._chips,
            run_number   = self.app.run_number,
            arch         = self.app.arch,
            project_dir  = self.app._project_dir,
            backend      = self._backend,
        )
    )

# After:
def _advance_to_run(self, chip_queues: list[list[dict]],
                    side_quest_pool: list[dict] | None = None) -> None:
    ...
    self.app.push_screen(
        RunScreen(
            chip_queues      = chip_queues,
            num_chips        = self._chips,
            run_number       = self.app.run_number,
            arch             = self.app.arch,
            project_dir      = self.app._project_dir,
            backend          = self._backend,
            side_quest_pool  = side_quest_pool or [],
        )
    )
```

- [ ] **Step 3: Update `RunScreen.__init__` to accept and store the pool (~line 893)**

```python
# Before:
def __init__(self, chip_queues: list[list[dict]], num_chips: int,
             run_number: int, arch: str, project_dir: Path,
             backend: str = "forge", **kwargs) -> None:
    super().__init__(**kwargs)
    self.chip_queues  = chip_queues
    ...

# After:
def __init__(self, chip_queues: list[list[dict]], num_chips: int,
             run_number: int, arch: str, project_dir: Path,
             backend: str = "forge",
             side_quest_pool: list[dict] | None = None,
             **kwargs) -> None:
    super().__init__(**kwargs)
    self.chip_queues  = chip_queues
    ...
    # Side quest state — populated from _build_side_quest_pool in curated mode
    self._side_quest_pool:     list[dict] = list(side_quest_pool or [])
    self._side_quest_chips:    set[int]   = set()
    self._side_quest_procs:    dict[int, asyncio.subprocess.Process] = {}
    self._rally_interrupt_flag: bool      = False
```

Add these four lines immediately after the existing `self._all_done: bool = False` line in `__init__`.

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('expedition_tui.py', doraise=True); print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add expedition_tui.py
git commit -m "Thread side_quest_pool from SetupScreen through to RunScreen"
```

---

## Task 3 — Dispatch integration: inject side quests and update RALLY quorum (`expedition_tui.py`)

**Files:**
- Modify: `expedition_tui.py` — `_dispatch_next()` and `_fire_rally()`

This is the core logic change. Two sub-problems:

### 3A. Inject side quests when idle + RALLY pending

At the end of `_dispatch_next`, before the pool-empty check, add:

```python
# Juggle while waiting for RALLY quorum: launch side quest on any free chip.
if (self._mesh_holding
        and self._free_chips
        and self._side_quest_pool
        and not self._rally_interrupt_flag):
    chip_id = min(self._free_chips)
    self._free_chips.discard(chip_id)
    self._side_quest_chips.add(chip_id)
    model = self._side_quest_pool.pop(0)
    self._launch_side_quest(chip_id, model)
    # Keep scanning — multiple chips may be free simultaneously.
    self._dispatch_next()
    return
```

Insert this block between "# Scan the pool for a dispatchable model" loop and the pool-empty check at the bottom. Exact location: after the `for i, model in enumerate(self._model_pool):` loop ends (the `continue` / closing of the for-loop) and before the `if not self._model_pool and not self._mesh_holding ...` line.

### 3B. Include side quest chips in RALLY quorum check

The existing RALLY quorum check at the top of `_dispatch_next` is:
```python
if self._mesh_holding:
    chips_needed: int = self._mesh_holding["chips_needed"]
    if len(self._free_chips) >= chips_needed:
        chip_ids = sorted(self._free_chips)[:chips_needed]
        self._fire_rally(self._mesh_holding, chip_ids)
        return
```

Replace with:
```python
if self._mesh_holding:
    chips_needed: int = self._mesh_holding["chips_needed"]
    # Count idle chips + interruptible side-quest chips toward quorum.
    all_available = self._free_chips | self._side_quest_chips
    if len(all_available) >= chips_needed:
        chip_ids = sorted(all_available)[:chips_needed]
        self._fire_rally(self._mesh_holding, chip_ids)
        return
```

### 3C. Kill side quest procs in `_fire_rally`

`_fire_rally` is already `@work async`. Add side quest cleanup at the top, right after setting `self._mesh_holding = None`:

```python
@work
async def _fire_rally(self, mesh_model: dict, chip_ids: list[int]) -> None:
    """Handle a RALLY event: show banner, fire mesh subprocess."""
    self._mesh_holding       = None
    self._opportunist_active = False
    self._rally_interrupt_flag = True   # signal side quest workers to exit

    # Kill any side quest subprocesses on the rally chips.
    # Side quest workers check _rally_interrupt_flag and exit on their own,
    # but we also send SIGKILL to guarantee prompt cleanup.
    for cid in list(chip_ids):
        proc = self._side_quest_procs.pop(cid, None)
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            self._side_quest_chips.discard(cid)
            try:
                el = self.query_one("#event-log", EventLog)
                el.write(f"[yellow]⚡ C{cid} SIDE QUEST cut short — ALL CHIPS CALLED TO RALLY[/]")
            except Exception:
                pass

    for cid in chip_ids:
        self._free_chips.discard(cid)
    # ... rest of existing _fire_rally body unchanged ...
```

- [ ] **Step 1: Update RALLY quorum check in `_dispatch_next`** (3B above)

- [ ] **Step 2: Add side quest injection block in `_dispatch_next`** (3A above, after the model_pool scan loop)

- [ ] **Step 3: Add side quest cleanup to `_fire_rally`** (3C above)

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('expedition_tui.py', doraise=True); print('ok')"
python3 -c "from expedition_tui import ExpeditionTUI; print('import ok')"
```

Expected: both `ok`.

- [ ] **Step 5: Commit**

```bash
git add expedition_tui.py
git commit -m "Dispatch: inject side quests on idle chips, include in RALLY quorum"
```

---

## Task 4 — `_launch_side_quest` method (`expedition_tui.py`)

**Files:**
- Modify: `expedition_tui.py` — add `_launch_side_quest` as a `@work` async method on RunScreen

Add immediately after `_launch_model` (after `_on_chip_free`):

```python
@work
async def _launch_side_quest(self, chip_id: int, model: dict) -> None:
    """Run a bonus model on an idle chip while waiting for RALLY quorum.

    Uses the same forge worker subprocess as _launch_model but with:
    - Separate tracking via _side_quest_chips / _side_quest_procs
    - Clean exit when _rally_interrupt_flag is set (RALLY takes priority)
    - Bonus EventLog messages instead of normal chip_done log
    """
    import time as _time
    start_t = _time.time()
    display  = model.get("display_name", model.get("model_id", "?").split("/")[-1])

    try:
        el = self.query_one("#event-log", EventLog)
        el.write(f"[cyan]⚡ C{chip_id} SIDE QUEST — {display}[/]")
    except Exception:
        pass

    try:
        panel = self.query_one(f"#chip-{chip_id}", ChipPanel)
        panel.write_line(f"\033[36m⚡ SIDE QUEST: {display}\033[0m\n")
    except Exception:
        panel = None

    # Write model JSON (separate filename from main worker to avoid clobbering).
    model_for_worker = {k: v for k, v in model.items() if k not in _WORKER_SKIP_KEYS}
    model_json_path  = f"/tmp/expedition_model_chip{chip_id}_sq.json"
    results_path     = f"/tmp/expedition_results_chip{chip_id}.csv"
    Path(model_json_path).write_text(json.dumps(model_for_worker))

    python_exe  = sys.executable
    worker_path = str(self._project_dir / "lib" / "expedition" / "expedition_worker.py")
    env = {
        **os.environ,
        "TT_VISIBLE_DEVICES":    str(chip_id),
        "TT_METAL_ARCH_NAME":    self.arch,
        "TT_METAL_LOGGER_LEVEL": "FATAL",
        "TT_MESH_GRAPH_DESC_PATH": str(
            self._project_dir / "mesh_graph_descriptors"
            / "p100_mesh_graph_descriptor.textproto"
        ),
        "PYTHONUNBUFFERED": "1",
    }

    proc = await asyncio.create_subprocess_exec(
        python_exe, worker_path,
        "--chip",       str(chip_id),
        "--run",        str(self.run_number),
        "--bestiary",   str(self._project_dir / "data" / "bestiary.json"),
        "--model-json", model_json_path,
        "--results",    results_path,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
    )
    self._side_quest_procs[chip_id] = proc

    async for raw in proc.stdout:
        if self._rally_interrupt_flag:
            break
        line = raw.decode("utf-8", errors="replace")
        if panel:
            panel.write_line(line)
        self._parse_for_events(chip_id, line)

    if self._rally_interrupt_flag:
        # _fire_rally owns this chip — don't touch state, let it manage cleanup.
        return

    await proc.wait()
    elapsed = _time.time() - start_t

    # Clean up proc tracking before freeing chip.
    self._side_quest_procs.pop(chip_id, None)
    self._side_quest_chips.discard(chip_id)

    if panel:
        panel.mark_done(proc.returncode == 0)

    if proc.returncode == 0:
        # Speed tier bonus label (cosmetic display only — actual pts from worker).
        if   elapsed < 10: speed_label = f"⚡ {elapsed:.1f}s — BLAZING"
        elif elapsed < 20: speed_label = f"⚡ {elapsed:.1f}s — fast"
        else:               speed_label = f"{elapsed:.1f}s"
        try:
            el = self.query_one("#event-log", EventLog)
            el.write(f"[bold cyan]⚡ C{chip_id} BONUS ★ {display} — {speed_label}[/]")
        except Exception:
            pass
    else:
        try:
            el = self.query_one("#event-log", EventLog)
            el.write(f"[dim]⚡ C{chip_id} SIDE QUEST FAIL — {display}[/]")
        except Exception:
            pass

    # Free chip — may trigger another side quest or RALLY quorum check.
    self._free_chips.add(chip_id)
    self._dispatch_next()
```

- [ ] **Step 1: Add `_launch_side_quest` method after `_on_chip_free`**

Insert the full method body above after `_on_chip_free` and before `_on_all_done`.

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import py_compile; py_compile.compile('expedition_tui.py', doraise=True); print('ok')"
python3 -c "from expedition_tui import ExpeditionTUI; print('import ok')"
```

- [ ] **Step 3: Verify dispatch flow in dry-run (no hardware needed)**

```bash
python3 -c "
from expedition_tui import RunScreen
from pathlib import Path
from expedition import _build_curated_queue
queues, pool = _build_curated_queue(4)
print('queues:', [len(q) for q in queues])
print('pool:', [m['display_name'] for m in pool])
# Construct RunScreen in isolation to verify __init__ accepts side_quest_pool
rs = RunScreen.__new__(RunScreen)
rs._side_quest_pool = list(pool)
rs._side_quest_chips = set()
rs._side_quest_procs = {}
rs._rally_interrupt_flag = False
print('RunScreen state initialized ok')
print('side_quest_pool has', len(rs._side_quest_pool), 'entries')
"
```

Expected: no errors, pool has 8 entries.

- [ ] **Step 4: Commit**

```bash
git add expedition_tui.py
git commit -m "Add _launch_side_quest: bonus forge compiles while waiting for RALLY"
```

---

## Task 5 — End-to-end verification (no hardware)

**Files:** Read-only verification pass.

- [ ] **Step 1: Full syntax + import check**

```bash
cd /home/ttuser/code/tt-forge-compiletron
python3 -c "
import py_compile
for f in ['expedition.py', 'expedition_tui.py']:
    py_compile.compile(f, doraise=True)
    print('syntax ok:', f)
from expedition_tui import ExpeditionTUI
print('import ok: ExpeditionTUI')
from expedition import _build_curated_queue, _build_side_quest_pool
print('import ok: builders')
"
```

- [ ] **Step 2: Verify curated queue + pool**

```bash
python3 -c "
from expedition import _build_curated_queue, _build_side_quest_pool
queues, pool = _build_curated_queue(4)
print('QUEUES:')
for i, q in enumerate(queues):
    for m in q:
        print(f'  C{i}: {m[\"display_name\"]:25} mesh={m[\"mesh_chips\"]}')
print()
print('SIDE QUEST POOL:')
for m in pool:
    print(f'  {m[\"display_name\"]:25} {m[\"model_id\"]}')
"
```

Expected:
- BLOOM shows `mesh=4`
- AlexNet/GPT-2/BEiT/DenseUNet show `mesh=1`
- Pool has 8 entries, all `mesh_chips=1`

- [ ] **Step 3: Verify all pool loader modules resolve**

```bash
python3 -c "
import sys, types, importlib.util
forgems_path = '/home/ttuser/code/tt-forge-models'
pkg = types.ModuleType('_forgems')
pkg.__path__ = [forgems_path]
sys.modules['_forgems'] = pkg
from expedition import _build_side_quest_pool
pool = _build_side_quest_pool(4)
for m in pool:
    mod = m['loader_module']
    mod_sub = mod.replace('_forgems.', '').replace('.', '/')
    path = f'{forgems_path}/{mod_sub}.py'
    try:
        spec = importlib.util.spec_from_file_location(mod, path)
        lm = importlib.util.module_from_spec(spec)
        sys.modules[mod] = lm
        spec.loader.exec_module(lm)
        loader = lm.ModelLoader()
        print(f'OK  {m[\"display_name\"]}')
    except Exception as e:
        print(f'ERR {m[\"display_name\"]}: {e}')
" 2>/dev/null
```

Expected: all 8 show `OK`.

- [ ] **Step 4: Commit**

```bash
git add -p  # review and stage only if needed
git commit -m "feat: juggle-while-we-wait side quest speed run on idle chips" --allow-empty
```

---

## Key invariants to preserve

1. **RALLY always fires** — even if all chips are doing side quests. The quorum check `free | side_quest_chips >= chips_needed` guarantees this.
2. **No double-free** — `_launch_side_quest` returns immediately (without touching chip state) if `_rally_interrupt_flag` is True, because `_fire_rally` already owns those chips.
3. **Watchdog is safe** — `_watchdog_check` only triggers when `not self._mesh_holding`, which is only true after RALLY fires. Side quests run while `_mesh_holding` is set.
4. **Non-curated mode unaffected** — `side_quest_pool` defaults to `[]`; no side quests are injected.
5. **Pool exhaustion is benign** — if all 8 side quest models are used before RALLY fires (unlikely), chips go idle normally. No crash.
