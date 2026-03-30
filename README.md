# TT-Forge Compiletron

Runs TT-Forge model compilation demos across Tenstorrent hardware.
Spawns a 4-pane tmux grid — one per chip — with live ASCII progress bars,
rotating figlet banners, and a victory celebration when all models finish.

<img width="3840" height="2002" alt="tt-forge-compiletron demo" src="https://github.com/user-attachments/assets/3e93d7d6-8e02-49f6-92cb-e2d93c6caec2" />

**Tested on:** 4× P300C Blackhole chips &nbsp;|&nbsp; **101 models** across 15+ architectures &nbsp;|&nbsp; **94.4% success rate**

---

## Quick start

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron
pip install -r requirements.txt
source ~/tt-forge-fe/env/activate

bash scripts/run_4way_tmux.sh       # launches the 4-chip demo
```

See [INSTALL.md](INSTALL.md) for Docker setup and detailed prerequisites.

---

## CLI reference

```bash
python3 compiletron.py detect                   # detect hardware (uses tt-smi)
python3 compiletron.py models list              # all 101 models
python3 compiletron.py models families          # grouped by architecture
python3 compiletron.py models quick             # fastest 5 (good for smoke tests)
python3 compiletron.py models info ResNet-50    # details for one model
python3 compiletron.py models estimate --count 50 --chips 4

python3 compiletron.py run --quick              # compile 5 fastest models
python3 compiletron.py run --chip 0 --family resnet
python3 compiletron.py run --parallel           # launches run_4way_tmux.sh

python3 compiletron.py results                  # view saved results
python3 compiletron.py results report --output report.md
```

---

## How it works

`scripts/run_4way_tmux.sh` opens a tmux session with this layout:

```
┌──────────────┬──────────────
│  Chip 0      │  Chip 1
├──────────────┼──────────────
│  Chip 2      │  Chip 3
├──────────────┴──────────────
│  [████░░] progress per chip
```

Each pane runs `lib/worker.py` independently. Models are distributed
round-robin: chip N compiles models N, N+4, N+8, … The bottom strip
shows live `█░` progress bars for all four chips (updated every second).

**Auto-detect**: the script prefers native if `~/tt-forge-fe/env/activate`
exists, otherwise falls back to Docker.

---

## Model library

101 models in `lib/models.py`, spanning:

| Family | Count | Compile time |
|---|---|---|
| RegNet (X/Y) | 15 | 2–8s |
| VGG | 8 | 1–4s |
| EfficientNet B0–B7 | 8 | 3–25s |
| Swin Transformer | 6 | 18–45s |
| ResNet | 5 | 3–15s |
| DenseNet | 4 | 40–116s |
| ViT | 4 | 20–50s |
| ConvNeXt | 4 | 20–45s |
| + MobileNet, MNASNet, ResNeXt, SqueezeNet, AlexNet, … | | |

Add a model by appending a tuple to `MODEL_LIST` in `lib/models.py`:

```python
("MyModel", "family", lambda: my_loader(), (1, 3, 224, 224), "notes",
 {'time': 10.0, 'success': 1.0, 'params': '25M', 'complexity': 'medium'}),
```

---

## Project layout

```
compiletron.py          CLI entry point
lib/
  worker.py             per-chip compilation worker (visual pipeline)
  models.py             MODEL_LIST — 101 models with metadata
  hardware.py           tt-smi hardware detection
  discovery.py          scan Forge repos / HuggingFace for new models
scripts/
  run_4way_tmux.sh      4-chip tmux orchestrator (native + docker)
  status_display.sh     renders bottom progress bar strip
  docker/               Docker-mode worker scripts
  examples/             ready-to-run workflow scripts
docs/
  FORGE_SETUP.md        build tt-forge-fe from source
  MULTI_CHIP.md         round-robin distribution details
  MODEL_LIBRARY.md      full model catalog
  CONTAINER_USAGE.md    Docker usage guide
  DOCKER_REFERENCE.md   Docker build reference
  PARALLEL_4CHIP_GUIDE.md  4-chip setup walkthrough
tests/                  29 unit tests (no hardware required)
```

---

## Testing

```bash
./run_tests.sh
# or
python3 -m pytest tests/ -v -p no:asyncio
```

All 29 tests pass without hardware (uses mock tt-smi data).

---

## Source

Extracted from `~/tt-forge-creative-demos/` — original 4-chip sweep
with 102/108 models passing on 2026-03-21.
