# Installation

## Native (recommended)

Requires `tt-forge-fe` built and installed at `~/tt-forge-fe`.

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron

# Activate forge environment first, then install compiletron deps
source ~/tt-forge-fe/env/activate
pip install -r requirements.txt
```

Launch the TUI:

```bash
python3 expedition.py run --tui
```

### XLA backend (optional)

One-time setup for the JAX/PJRT backend:

```bash
python3 -m venv xla-venv
xla-venv/bin/pip install pjrt-plugin-tt jax==0.7.1 jaxlib==0.7.1 \
    flax==0.8.5 "transformers<5.0" torch \
    --index-url https://pypi.tenstorrent.com/simple/
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
| tt-smi | hardware detection |
| textual, pyfiglet, etc. | installed via `requirements.txt` |

---

## Verify the install

```bash
./run_tests.sh                              # all tests, no hardware required
python3 expedition.py run --tui --limit 3  # dry run with TUI
```
