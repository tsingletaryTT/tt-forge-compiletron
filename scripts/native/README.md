# Native Scripts (No Docker)

These scripts run TT-Forge compilation **directly on the host** without Docker containers.

## Prerequisites

- TT-Forge installed on host system
- Python environment with forge, torch, etc.
- Tenstorrent hardware access
- `TT_METAL_HOME` and related environment variables set

## Scripts

### `run_parallel.sh`
Launches multiple Python processes for parallel compilation on host.

**Usage:**
```bash
./run_parallel.sh
```

**Features:**
- Direct hardware access (no Docker overhead)
- Uses host's forge installation
- Suitable for development and debugging

### `view_logs.sh`
Helper script to view compilation logs.

**Usage:**
```bash
./view_logs.sh [log_directory]
```

## When to Use Native vs Docker

### Use Native Scripts When:
- ✅ You're developing/debugging forge itself
- ✅ You need maximum performance (no container overhead)
- ✅ You want to test local code changes immediately
- ✅ Docker isn't available on your system

### Use Docker Scripts When:
- ✅ You want isolation between compilation jobs
- ✅ You need reproducible environments
- ✅ You're running on shared/production hardware
- ✅ You want to avoid polluting host environment

## Environment Setup

Native scripts require these environment variables:

```bash
export TT_METAL_HOME=/path/to/tt-metal
export ARCH_NAME=blackhole
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
# ... (see tt-metal/tt-forge-fe documentation)
```

## See Also

- `../docker/` - Docker-based scripts for containerized execution
- `~/tt-forge-creative-demos/` - Original parallel execution examples
- `../../README.md` - Main project documentation
