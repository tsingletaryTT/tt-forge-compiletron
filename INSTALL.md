# Installation

The quickest path is the smart installer — it checks your environment,
detects common gotchas (wrong `forge` package, stale shm segments, missing
mesh descriptor), compares installed versions against TT PyPI, and calls
`setup-venvs.sh` to fix anything broken:

```bash
bash scripts/install.sh          # check + fix everything
bash scripts/install.sh --status # check only, no installs
```

For manual step-by-step instructions, source-build options, and
hardware-specific notes, see the sections below.

---

Compiletron needs two Python 3.12 venvs:

| Venv | Path | Purpose |
|---|---|---|
| forge backend | `~/tt-forge-venv` | `forge.compile()` — PyTorch → TT silicon |
| XLA backend | `~/tt-xla/venv` | `pjrt-plugin-tt` — JAX/Flax → TT silicon |

Both are installed from **pre-built pip wheels** on Tenstorrent's package index.
No source build required.

---

## Quick path — one script, both backends

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron

bash scripts/setup-venvs.sh
```

The script:

1. Installs Python 3.12, build tools, and tmux if missing (via `apt`)
2. Sets hugepages to 128 and persists the setting across reboots
3. Creates `~/tt-forge-venv` and installs `forge` + compiletron deps
4. Writes a `~/tt-forge-fe/env/activate` shim so the harness finds the venv
5. Creates `~/tt-xla/venv` and installs `pjrt-plugin-tt` + JAX + Flax stack
6. Links the bundled mesh descriptor if `~/tt-xla` is a pip-wheel install

**Individual backend only:**

```bash
bash scripts/setup-venvs.sh --forge   # forge backend only
bash scripts/setup-venvs.sh --xla     # XLA backend only
```

---

## After setup

```bash
# Activate forge env and run
source ~/tt-forge-fe/env/activate
python3 expedition.py run --tui

# XLA backend (uses ~/tt-xla/venv automatically — no manual activation needed)
python3 expedition.py run --tui --backend xla

# Forge + XLA side by side on alternating chips
python3 expedition.py run --tui --backend mixed
```

---

## Manual install (step by step)

### 1. System packages

```bash
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev \
    build-essential git tmux
```

### 2. Hugepages

tt-metal requires hugepages for device memory mapping.

```bash
sudo sysctl -w vm.nr_hugepages=128

# Persist across reboots:
echo 'vm.nr_hugepages=128' | sudo tee /etc/sysctl.d/99-tenstorrent-hugepages.conf
```

Verify: `cat /proc/sys/vm/nr_hugepages` should be ≥ 64.

### 3. tt-smi

```bash
pip install tt-smi --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

Verify: `tt-smi -s` (snapshot mode) returns JSON.

### 4. Forge backend venv

```bash
python3.12 -m venv ~/tt-forge-venv

~/tt-forge-venv/bin/pip install \
    forge \
    --extra-index-url https://pypi.eng.aws.tenstorrent.com/

~/tt-forge-venv/bin/pip install -r requirements.txt

# Optional but needed for FlagEmbedding seed models:
~/tt-forge-venv/bin/pip install FlagEmbedding tf-keras
```

Verify: `~/tt-forge-venv/bin/python -c "import forge; print(forge.__version__)"`

#### 4a. Write the activate shim

The expedition harness calls `source ~/tt-forge-fe/env/activate` to set up the
forge environment. If you installed via pip wheel (not a source build), write
this shim so that path resolves correctly:

```bash
mkdir -p ~/tt-forge-fe/env
cat > ~/tt-forge-fe/env/activate << 'EOF'
#!/usr/bin/env bash
export TTFORGE_TOOLCHAIN_DIR="${TTFORGE_TOOLCHAIN_DIR:-${HOME}/tt-forge-venv}"
export TTFORGE_PYTHON_VERSION="${TTFORGE_PYTHON_VERSION:-python3.12}"
export TTFORGE_VENV_DIR="${TTFORGE_VENV_DIR:-${HOME}/tt-forge-venv}"
export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-${HOME}/tt-forge-venv}"
export TTMLIR_VENV_DIR="${TTMLIR_VENV_DIR:-${HOME}/tt-forge-venv}"
export TTMLIR_ENV_ACTIVATED=1
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
source "${HOME}/tt-forge-venv/bin/activate"
EOF
chmod +x ~/tt-forge-fe/env/activate
```

> **`ARCH_NAME`** defaults to `blackhole` (P150/P300 Blackhole boards).
> Override before sourcing for other hardware:
> - Wormhole n150/n300: `export ARCH_NAME=wormhole_b0`
> - Grayskull e75/e150: `export ARCH_NAME=grayskull`

### 5. XLA backend venv

```bash
mkdir -p ~/tt-xla
python3.12 -m venv ~/tt-xla/venv

~/tt-xla/venv/bin/pip install \
    pjrt-plugin-tt \
    jax jaxlib \
    flax \
    "transformers<5.0" \
    torch torchvision timm \
    pyfiglet \
    --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

> **`torchvision` and `timm`** — several tt-forge-models JAX loaders import from
> `tools/utils.py` which pulls these in at module level. Without them every JAX
> seed model fails with `No module named 'torchvision'`.
>
> **`pyfiglet`** — ASCII art banners in the XLA worker. Without it banners fall
> back to plain text (harmless but less dramatic).

Verify: `~/tt-xla/venv/bin/python -c "import jax; import pjrt_plugin_tt; print('ok')"`

#### 5a. Mesh graph descriptor

The XLA worker needs a mesh descriptor for P300/P150 Blackhole boards. It
auto-detects from `~/tt-xla` if you have a full tt-xla source build there. For
pip-wheel installs, link the bundled descriptor:

```bash
MESH_DIR=~/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors
mkdir -p "$MESH_DIR"
ln -sf "$(pwd)/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto" \
    "$MESH_DIR/p100_mesh_graph_descriptor.textproto"
```

Or set the env var explicitly at runtime:

```bash
export TT_MESH_GRAPH_DESC_PATH=~/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
```

---

## Source build (optional)

If you need to develop or patch forge/tt-xla themselves, build from source:

### Forge (tt-forge-fe)

```bash
cd ~
git clone https://github.com/tenstorrent/tt-forge-fe.git
cd tt-forge-fe
./build.sh          # 45–90 min; creates /opt/ttforge-toolchain/venv
source env/activate
```

The real `env/activate` from the source tree sets `TTFORGE_TOOLCHAIN_DIR`,
activates `/opt/ttforge-toolchain/venv`, and sets `ARCH_NAME` automatically. The
pip-wheel shim (`~/tt-forge-fe/env/activate` written above) must be absent or
the source-build activate takes precedence.

### XLA (tt-xla)

```bash
cd ~
git clone https://github.com/tenstorrent/tt-xla.git
cd tt-xla
# Follow tt-xla build instructions from docs/src/getting_started.md
```

Source-built tt-xla places the mesh descriptor in the expected path automatically;
no symlink needed.

---

## Prerequisites summary

| Requirement | Notes |
|---|---|
| Ubuntu 24.04 | Tested platform |
| Python 3.12 | `python3.12 --version` |
| Hugepages ≥ 64 | Required by tt-metal; 128 recommended |
| tt-smi | Hardware detection (`tt-smi -s` for JSON snapshot mode) |
| tmux | Required by `scripts/run_expedition.sh` layout |
| `~/tt-forge-venv` | Forge backend; contains `forge` wheel |
| `~/tt-xla/venv` | XLA backend; contains `pjrt-plugin-tt` |
| `~/tt-forge-fe/env/activate` | Activation shim (written by setup-venvs.sh) |
| `~/code/tt-forge-models` | Seed model zoo (optional; frontier-only works without it) |

---

## Verify the install

```bash
# Run all tests (no hardware required)
./run_tests.sh

# Syntax-check main modules
python3 -c "
import py_compile
for f in ['expedition.py', 'expedition_tui.py',
          'lib/expedition/bestiary.py', 'lib/expedition/router.py',
          'lib/expedition/decoder.py']:
    py_compile.compile(f, doraise=True)
    print('ok', f)
"

# Dry run — TUI auto-starts in 4s, press Q on summary screen to exit
source ~/tt-forge-fe/env/activate
python3 expedition.py run --tui --seed-only --limit 3 --chips 1 --no-predownload
```

---

## Troubleshooting

**Workers crash immediately (SIGSEGV / "Segmentation fault")**
tt-metal requires hugepages. Check and set:

```bash
cat /proc/sys/vm/nr_hugepages          # should be ≥ 64 (128 recommended)
sudo sysctl -w vm.nr_hugepages=128
echo 'vm.nr_hugepages=128' | sudo tee /etc/sysctl.d/99-tenstorrent-hugepages.conf
```

Also clean stale shared-memory segments from crashed runs:

```bash
rm -f /dev/shm/sm_segment.tt-quietbox.*
tt-smi -r    # hardware device reset
```

**"forge not importable"**
Activate the forge env: `source ~/tt-forge-fe/env/activate`

**Forge workers segfault even after activating the env**
The toolchain Python in `/opt/ttforge-toolchain/venv/` may have a conflicting
partial `forge/` package. Try running with the forge venv Python directly:

```bash
~/tt-forge-venv/bin/python expedition.py run --tui
```

**XLA banners show plain text instead of ASCII art**

```bash
~/tt-xla/venv/bin/pip install pyfiglet
```

**XLA models fail with `No module named 'torchvision'`**

```bash
~/tt-xla/venv/bin/pip install torchvision timm
```

**XLA fails with `TT_FATAL: Custom fabric mesh graph descriptor path must be specified`**
The PJRT backend can't find the mesh descriptor for your board. Set it manually:

```bash
export TT_MESH_GRAPH_DESC_PATH=~/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
# Or re-run setup-venvs.sh --xla to write the symlink automatically.
```

**FlagEmbedding import fails with Keras conflict**

```bash
~/tt-forge-venv/bin/pip install tf-keras
```

**TUI hangs at "Waiting for chips..."**
The watchdog detects completion via status files. If it never transitions, clean
stale files from a previous crashed run:

```bash
rm -f /tmp/expedition_chip_*.status
```

**Cast playback is jerky**
Re-compress with `--min-gap 0.02`:

```bash
python3 scripts/compress_cast.py docs/demo_raw.cast docs/demo.cast \
    --max-idle 1.2 --min-gap 0.02
```

---

## Docker

Use Docker if you prefer a fully isolated environment. The image compiles
tt-metal and tt-forge-fe from source (~21 GB, 2–3 hour one-time build).

```bash
./docker-build-full.sh
./docker-run.sh run --tui --chips 4
```

See [docs/CONTAINER_DEPLOYMENT.md](docs/CONTAINER_DEPLOYMENT.md) for more.
