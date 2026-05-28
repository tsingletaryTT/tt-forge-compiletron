# Installation

## Native (recommended)

Requires `tt-forge-fe` built and installed at `~/tt-forge-fe`.

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron

# Activate forge environment first, then install compiletron deps
source ~/tt-forge-fe/env/activate
pip install -r requirements.txt

# Additional deps for embedding models (FlagEmbedding requires tf-keras shim)
pip install FlagEmbedding tf-keras
```

Launch the TUI:

```bash
python3 expedition.py run --tui
```

The Setup screen auto-starts after 4 seconds if you don't press Enter.

### XLA backend (optional)

One-time setup for the JAX/PJRT backend. Uses a separate virtualenv to
avoid conflicts with the forge environment. The convention used on this
machine is `~/tt-xla/venv`; adjust the path to match your tt-xla install:

```bash
XLA_VENV=~/tt-xla/venv   # adjust if your tt-xla venv lives elsewhere

$XLA_VENV/bin/pip install pjrt-plugin-tt jax jaxlib \
    flax "transformers<5.0" torch torchvision timm pyfiglet \
    --index-url https://pypi.tenstorrent.com/simple/
```

> **`torchvision` and `timm`** are required because several tt-forge-models JAX
> loaders import from `tools/utils.py` which pulls them in at module level.
> Without them every JAX seed model fails immediately with `No module named 'torchvision'`.

> **`pyfiglet`** is required for ASCII art model name banners in the XLA worker.
> Without it banners fall back to plain text (harmless).

The XLA worker also needs a mesh graph descriptor for P300/P150 boards. It
auto-detects this from inside the tt-xla tree — no manual step needed as long
as `~/tt-xla` is present. If you keep tt-xla at a different path, set:
```bash
export TT_MESH_GRAPH_DESC_PATH=/path/to/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
```

---

## Docker

Use Docker if you don't have a local Forge build. The image compiles
tt-metal and tt-forge-fe from source (~21 GB, 2–3 hour one-time build).

```bash
# One-time build
./docker-build-full.sh

# Launch TUI (requires hardware device)
./docker-run.sh run --tui --chips 4

# CLI run
./docker-run.sh run --chips 4 --limit 20
```

See [docs/CONTAINER_DEPLOYMENT.md](docs/CONTAINER_DEPLOYMENT.md) for more.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | `python3 --version` |
| tt-metal | built at `~/tt-metal` |
| tt-forge-fe | built at `~/tt-forge-fe` — must be built from source |
| tt-smi | hardware detection (`tt-smi -s` for JSON snapshot mode) |
| Hugepages ≥ 64 | required by tt-metal: `sudo sysctl -w vm.nr_hugepages=128` |
| textual, pyfiglet, etc. | installed via `requirements.txt` |
| FlagEmbedding | `pip install FlagEmbedding tf-keras` (embedding models) |
| torchvision + timm in XLA venv | required by JAX loaders (tools/utils.py imports them) |
| pyfiglet in xla-venv | needed for ASCII banners in XLA worker |

---

## Verify the install

```bash
./run_tests.sh                                      # all tests, no hardware required

# Syntax check all main modules
python3 -c "
import py_compile
for f in ['expedition.py', 'expedition_tui.py',
          'lib/expedition/bestiary.py', 'lib/expedition/router.py',
          'lib/expedition/decoder.py']:
    py_compile.compile(f, doraise=True)
    print('ok', f)
"

# Dry run with TUI (auto-starts in 4s, press Q on summary to exit)
python3 expedition.py run --tui --seed-only --limit 3 --chips 1 --no-predownload
```

---

## Troubleshooting

**Workers crash immediately (SIGSEGV / "Segmentation fault")** — tt-metal requires
hugepages for device memory mapping. Check and set:
```bash
cat /proc/sys/vm/nr_hugepages          # should be ≥ 64 (128 recommended)
sudo sysctl -w vm.nr_hugepages=128     # set for current boot
# To persist across reboots:
echo 'vm.nr_hugepages=128' | sudo tee /etc/sysctl.d/99-hugepages.conf
```
Also clean any stale shared-memory segments left by previous crashes:
```bash
rm -f /dev/shm/sm_segment.tt-quietbox.*
tt-smi -r    # hardware device reset
```

**"forge not importable"** — activate the forge env: `source ~/tt-forge-fe/env/activate`

**Forge workers segfault even after activating forge env** — the toolchain Python in
`/opt/ttforge-toolchain/venv/` may have a conflicting partial `forge/` package that
prevents `_C.so` from loading. Try running expedition with the system Python instead:
```bash
/usr/bin/python3 expedition.py run --tui
# or with the tenstorrent venv if present:
~/.tenstorrent-venv/bin/python3 expedition.py run --tui
```

**XLA banners show plain text instead of ASCII art** — install pyfiglet in the XLA venv:
```bash
~/tt-xla/venv/bin/pip install pyfiglet
```

**XLA models all fail with `No module named 'torchvision'`** — torchvision and timm
are not installed in the XLA venv. Install them:
```bash
~/tt-xla/venv/bin/pip install torchvision timm
```

**XLA fails with `TT_FATAL: Custom fabric mesh graph descriptor path must be specified`**
— the PJRT backend can't find the mesh descriptor for your P300/P150 board. The worker
auto-sets this from `~/tt-xla` on startup. If `~/tt-xla` doesn't exist, set manually:
```bash
export TT_MESH_GRAPH_DESC_PATH=~/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
```

**FlagEmbedding import fails with Keras conflict** — install the tf-keras shim:
```bash
pip install tf-keras
```

**TUI hangs at "Waiting for chips..."** — the watchdog timer will detect completion
within 2 seconds via status files. If it never transitions, check for stale
`/tmp/expedition_chip_*.status` files from a previous crashed run and delete them.

**Cast playback is jerky** — re-compress with `--min-gap 0.02` to floor inter-event
gaps to 20 ms:
```bash
python3 scripts/compress_cast.py docs/demo_raw.cast docs/demo.cast \
    --max-idle 1.2 --min-gap 0.02
```
