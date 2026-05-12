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
avoid conflicts with the forge environment:

```bash
python3 -m venv xla-venv
xla-venv/bin/pip install pjrt-plugin-tt jax==0.7.1 jaxlib==0.7.1 \
    flax==0.8.5 "transformers<5.0" torch pyfiglet \
    --index-url https://pypi.tenstorrent.com/simple/
```

> **Note:** `pyfiglet` is required in the XLA venv to display ASCII art model
> name banners during compilation. Without it the banners fall back to plain text.

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

**XLA banners show plain text instead of ASCII art** — install pyfiglet in xla-venv:
```bash
xla-venv/bin/pip install pyfiglet
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
