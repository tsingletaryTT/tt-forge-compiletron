# Docker-Based Scripts

These scripts run TT-Forge compilation **inside Docker containers** for isolation and portability.

## Prerequisites

- Docker installed and user in `docker` group
- TT-Forge Docker image built: `tt-forge-compiletron:minimal`
- Tenstorrent hardware access: `/dev/tenstorrent`

## Scripts

### `run_parallel_4chip.sh`
Launches 4 Docker containers simultaneously (one per chip) for parallel compilation.

**Usage:**
```bash
./run_parallel_4chip.sh test_name
```

**Features:**
- Each container isolated to single chip via `TT_VISIBLE_DEVICES`
- Automatic log collection to `/tmp/forge_parallel_*/`
- Exit status tracking for all 4 chips
- Parallel execution with no interference

**Example:**
```bash
cd /home/ttuser/code/tt-forge-compiletron
./scripts/docker/run_parallel_4chip.sh my_test
```

### `demo_4way_tmux.sh`
Creates a 2x2 tmux grid showing all 4 chips compiling in real-time.

**Usage:**
```bash
./demo_4way_tmux.sh test_name
tmux attach -t forge_demo
# Press Enter in each pane to start
```

**Features:**
- Visual monitoring of all 4 chips
- Side-by-side log viewing
- Great for demos and debugging
- tmux controls: `Ctrl+B` then arrow keys to switch panes

### `forge_worker.py`
Python worker script that runs inside each Docker container.

**Features:**
- Imports forge and compiles a test model
- Logs progress with timestamps and chip ID
- Returns exit code (0=success, 1=error)

## Current Status

✅ **Working chips**: 0, 2 (compile successfully)
⚠️ **Problematic chips**: 1, 3 (fabric control_plane errors)

This is due to ethernet topology configuration when isolating individual chips. Chips 0 & 2 work reliably for parallel testing.

## Troubleshooting

### Permission Denied
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

### Container Name Conflicts
```bash
docker ps -a | grep forge_chip | awk '{print $1}' | xargs -r docker rm -f
```

### Chip Reset Needed
```bash
tt-smi -r
sleep 3
```

## See Also

- `../native/` - Non-Docker scripts for direct hardware access
- `../../Dockerfile.minimal` - Docker image definition
- `../../PARALLEL_4CHIP_GUIDE.md` - Detailed parallel execution guide
