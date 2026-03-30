# Installation

## Native (recommended)

Requires `tt-forge-fe` built and installed at `~/tt-forge-fe`.

```bash
git clone git@github.com:tsingletaryTT/tt-forge-compiletron.git
cd tt-forge-compiletron

pip install -r requirements.txt

source ~/tt-forge-fe/env/activate
python3 compiletron.py detect        # confirm hardware is visible
```

Run the full 4-chip demo:

```bash
bash scripts/run_4way_tmux.sh
```

The script auto-detects native vs Docker. It prefers native if `~/tt-forge-fe/env/activate` exists.

---

## Docker

Use Docker if you don't have a local Forge build. The image compiles tt-metal and tt-forge-fe
from source (~21 GB, 2–3 hour one-time build) so every chip is fully isolated.

```bash
# One-time build (grab a coffee)
./docker-build-full.sh

# Test a single chip
docker run --rm --device=/dev/tenstorrent:/dev/tenstorrent \
    --shm-size=16g -e TT_VISIBLE_DEVICES=0 \
    tt-forge-compiletron:full \
    python3 /app/scripts/docker/forge_worker.py test

# 4-way parallel via tmux
bash scripts/run_4way_tmux.sh --mode docker
```

See [docs/CONTAINER_DEPLOYMENT.md](docs/CONTAINER_DEPLOYMENT.md) and
[docs/DOCKER_REFERENCE.md](docs/DOCKER_REFERENCE.md) for more detail.

---

## Dependencies

| Requirement | Notes |
|---|---|
| Python 3.12 | `python3 --version` |
| tt-metal | `~/tt-metal` |
| tt-forge-fe | `~/tt-forge-fe` — must be built from source |
| tt-smi | for hardware detection |
| pyfiglet | installed via `requirements.txt` |

---

## Verify the install

```bash
./run_tests.sh          # 29 tests, no hardware required
python3 compiletron.py detect
python3 compiletron.py models list
```
