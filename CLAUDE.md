# tt-forge-compiletron — project notes for Claude

## What this project is
Compiletron is a TUI expedition framework that mass-compiles real HuggingFace and tt-forge-models on Tenstorrent hardware using forge (PyTorch) and XLA backends. Results are logged to a bestiary (data/bestiary.json) and displayed on the website (landing page).

## Bestiary — keep it real and growing
**Always update `data/bestiary.json`** with real compile results from actual hardware runs.
- Wins AND errors both go in — failed models are honest proof points, not embarrassments.
- Every model that compiles (or fails) on real silicon is unofficial proof-of-hardware: "we ran this on Tenstorrent chips, here's what happened."
- The bestiary feeds the website landing page (bestiary showcase section). Adding new entries = more models shown as proof points.
- When adding a model to the curated demo queue or side quest pool, run it once to get a real result and update the bestiary entry.
- Don't fabricate compile times or scores — the bestiary is evidence.

## Curated demo queue (`_build_curated_queue`)
- Hand-picked models for the showcase recording: AlexNet (C0), GPT-2 (C1), BEiT (C2), DenseUNet FAIL (C3), BLOOM JAX 4-chip finale.
- The FAIL is intentional — shows real hardware behavior, not cherry-picked wins.
- Curated pool for "juggle while we wait" side quests: MobileNetV2, GhostNet, GoogleNet, EfficientNet-Lite, DenseNet, etc.

## Recording
- `bash scripts/record_demo.sh --curated` to record.
- `ROWS=50` is intentional — terminal is 54 rows, keeps 4-row breathing room.
- After recording: `python3 scripts/compress_cast.py docs/demo_raw.cast docs/demo.cast`

## BLOOM JAX priming
Before any curated run: BLOOM JAX weights must be converted from PyTorch and saved as a
Flax checkpoint. This only needs to happen once (or after clearing the cache).

```bash
JAX_PLATFORMS=cpu ~/tt-xla/venv/bin/python3 -c "
import sys, types, pathlib
forgems = types.ModuleType('_forgems'); forgems.__path__ = ['/home/ttuser/code/tt-forge-models']
sys.modules['_forgems'] = forgems

# Load with from_pt=True on CPU (avoids TT hardware Flax init crash)
from transformers import FlaxBloomForCausalLM
model = FlaxBloomForCausalLM.from_pretrained('bigscience/bloom-1b1', from_pt=True)

# Save as Flax msgpack — loader checks here before using from_pt=True
cache = pathlib.Path.home() / '.cache' / 'tt-forge-compiletron-flax' / 'bloom-1b1'
cache.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(cache))
print('Saved Flax checkpoint to', cache)
print('Files:', list(cache.iterdir()))
"
```

This saves `flax_model.msgpack` + config to `~/.cache/tt-forge-compiletron-flax/bloom-1b1/`.
On subsequent runs the BLOOM loader loads from there directly — no `from_pt=True`, no conflict
with the XLA worker's `_do_init=False` patch.

## SIGSEGV isolation
`forge.compile()` crashes the host process on some models. The forge worker runs each model in a subprocess with `FORGE_PYTORCH_ONLY=1` to isolate segfaults.

## Stale /dev/shm segments
After a crashed run: `find /dev/shm -name 'sm_segment.tt-quietbox.*.0' -delete` before re-running.
