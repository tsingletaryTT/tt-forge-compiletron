# Docker Full-Build Reference

This Docker image compiles **tt-metal** and **tt-forge-fe from source** to enable reliable 4-way parallel compilation with single-chip isolation.

## Quick Facts

- **Approach:** Full source build (tt-metal + tt-forge-fe)
- **Image Size:** ~21GB
- **Build Time:** 2-3 hours (first build, cached thereafter)
- **Result:** **4/4 chips compile successfully in parallel** (~6 seconds each)

## Why Full Source Build?

### The Problem with Pre-built Images

Pre-built `tt-forge-slim` images from `ghcr.io/tenstorrent/tt-forge-slim` have a **topology discovery bug** that prevents single-chip isolation:

```
ERROR: Physical chip id 1 not found in control plane chip mapping
n_log=1, n_phys=2  # Expects 1 chip, finds 2 - mismatch causes crash
```

- **Chips 1 & 3:** Control plane crashes immediately
- **Chips 0 & 2:** Hang at topology discovery
- **Root cause:** Forge-bundled tt-metal incompatible with STRICT mesh graph descriptors
- **TT_VISIBLE_DEVICES:** Not working with pre-built images

### The Solution

Build tt-metal and tt-forge-fe from source using:
- **Standalone tt-metal** (not forge-bundled version)
- **STRICT policy mesh graph descriptors** (for single-chip isolation)
- **Matched library versions** (no ABI mismatches)

**Result:** All 4 chips work independently in parallel containers.

## Build Process

### Stage 1: Builder (Compilation)

```dockerfile
FROM ubuntu:24.04

# Install build dependencies
RUN apt-get install build-essential cmake ninja-build clang-17 ...

# Clone and build tt-metal
RUN git clone https://github.com/tenstorrent/tt-metal.git
RUN cmake -B build && cmake --build build

# Clone and build tt-forge-fe
RUN git clone https://github.com/tenstorrent/tt-forge-fe.git
RUN cmake --build build -- install_ttforge
```

**Time:** 2-3 hours
**Cached:** Yes - subsequent builds reuse these layers

### Stage 2: Runtime (Deployment)

```dockerfile
FROM ubuntu:24.04

# Copy compiled artifacts from builder
COPY --from=builder /build/tt-metal /tt-metal
COPY --from=builder /build/tt-forge-fe /tt-forge-fe
COPY --from=builder /opt/ttforge-toolchain /opt/ttforge-toolchain

# Add runtime dependencies
RUN apt-get install libncurses6 libopenmpi3t64 openmpi-bin ...

# Copy application
COPY lib/ scripts/ mesh_graph_descriptors/ ...
```

**Time:** ~5 minutes (mostly Python package installs)

## Image Contents

### Compiled from Source
- **tt-metal:** Complete build with UMD, device drivers
- **tt-forge-fe:** Full Forge compiler with MLIR/TVM
- **Toolchain:** Python 3.12 venv with all dependencies

### Runtime Dependencies
- `libncurses6` - Terminal library
- `libopenmpi3t64` + `openmpi-bin` - MPI runtime and tools
- System libraries for hardware access

### Application Files
- Compiletron Python application
- Model library (101 models)
- Worker scripts for parallel execution
- **Mesh graph descriptors** (STRICT policy for single-chip)

## Build Instructions

```bash
# Build image (2-3 hours, one-time)
docker build -t tt-forge-compiletron:full .

# Check image size
docker images tt-forge-compiletron:full
# OUTPUT: ~21GB content size
```

### Build Options

**Speed up compilation:**
```bash
# Use more cores (default: auto-detected)
docker build --build-arg MAKE_JOBS=16 -t tt-forge-compiletron:full .
```

**Use cache from previous build:**
```bash
# Subsequent builds use cached layers
docker build -t tt-forge-compiletron:full .
# Only runtime layers rebuild (~5 minutes)
```

## Usage

### Single Chip Test

```bash
docker run --rm \
    --device=/dev/tenstorrent:/dev/tenstorrent \
    --shm-size=16g \
    -e TT_VISIBLE_DEVICES=0 \
    -e TT_METAL_ARCH_NAME=blackhole \
    -e TT_MESH_GRAPH_DESC_PATH=/app/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto \
    tt-forge-compiletron:full \
    python3 /app/scripts/docker/forge_worker.py test_single

# Expected output:
# ✓ Compilation completed in ~6s
# ✓ Worker finished successfully!
```

### 4-Way Parallel

```bash
./scripts/docker/run_parallel_4chip.sh

# Expected output:
# Chip 0: 5.87s ✓
# Chip 1: 5.74s ✓
# Chip 2: 6.21s ✓
# Chip 3: 6.12s ✓
# 🎉 ALL 4 CHIPS COMPILED SUCCESSFULLY!
```

## Docker Flags Explained

### Required Flags

**`--device=/dev/tenstorrent:/dev/tenstorrent`**
- Maps Tenstorrent hardware device into container
- Required for UMD to access chips

**`--shm-size=16g`**
- Shared memory for UMD sysmem allocation
- Default 64MB too small, needs ~16GB minimum
- Host has 125GB, container gets portion

**`-e TT_VISIBLE_DEVICES=N`**
- Isolates container to single chip
- Critical for parallel execution
- Each container sees only one chip

### Optional Flags (Not Needed)

**`--ipc=host`** - NOT required (isolated IPC works)
**`--privileged`** - NOT required (device mapping sufficient)

### Environment Variables

```bash
TT_VISIBLE_DEVICES=0               # Chip to use (0-3)
TT_METAL_ARCH_NAME=blackhole       # Architecture
TT_MESH_GRAPH_DESC_PATH=...        # Path to mesh descriptor (STRICT policy)
TT_METAL_HOME=                     # Clear forge-bundled path
TT_METAL_VERSION=                  # Use compiled version
```

## Troubleshooting

### Build Fails

**Error: "No space left on device"**
```bash
# Check Docker storage
docker system df

# Clean up old images
docker system prune -a

# Need ~50GB free for build
```

**Error: "Killed" during compilation**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory → 16GB+
```

### Runtime Fails

**Error: "libncurses.so.6: cannot open shared object"**
```bash
# Runtime dependencies missing - rebuild image
# This was fixed in final Dockerfile
```

**Error: "Mesh graph descriptor not found"**
```bash
# mesh_graph_descriptors/ not in image - rebuild
# This was fixed in final Dockerfile
```

**Error: "MPI_Init failed"**
```bash
# openmpi-bin not installed - rebuild
# This was fixed in final Dockerfile
```

## Fixes Applied (Build Iteration History)

1. **libncurses6** - Added missing terminal library
2. **libopenmpi3t64 + openmpi-bin** - Added MPI runtime + tools
3. **LD_LIBRARY_PATH** - Added tt-mlir library paths
4. **PATH** - Put venv/bin first (use toolchain Python)
5. **tt_tvm reinstall** - Fixed editable install paths (builder → runtime)
6. **mesh_graph_descriptors/** - Copied STRICT policy descriptors
7. **Toolchain venv pip** - Use venv pip (avoid PyTorch conflicts)

**Total rebuilds:** 10
**Final result:** 4/4 chips working ✅

## Future Plans

### Plugin-Based API (When Stable)

```python
# Future approach (not yet working)
import torch
import torch_plugin_tt

model = torch.nn.Linear(128, 128)
compiled = torch.compile(model, backend="tt")
output = compiled(inputs)
```

**Benefits:**
- Smaller images (~10-15GB vs 21GB)
- Use PyPI packages instead of source build
- Faster build times (~15 minutes vs 2-3 hours)

**Blockers (as of 2026-03-26):**
- Circular import bugs in `torch_plugin_tt`
- Breaking API changes in dev releases
- Not yet stable enough for production

**Timeline:** TBD (follow forge releases)

## Comparison

| Aspect | Full Build (Current) | Plugin API (Future) |
|--------|---------------------|---------------------|
| Image Size | 21GB | 10-15GB |
| Build Time | 2-3 hours | 15 minutes |
| Approach | Compile from source | PyPI packages |
| Stability | ✅ Working | ⚠️  In development |
| 4-way Parallel | ✅ 4/4 chips | 🔮 TBD |
| Maintenance | High (source tracking) | Low (version pins) |

## Production Deployment

### Build Once, Deploy Everywhere

```bash
# On build server (one time)
docker build -t registry.company.com/tt-forge-compiletron:v1.0 .
docker push registry.company.com/tt-forge-compiletron:v1.0

# On worker nodes
docker pull registry.company.com/tt-forge-compiletron:v1.0

# Run 4-way parallel
for chip in {0..3}; do
    docker run -d \
        --device=/dev/tenstorrent:/dev/tenstorrent \
        --shm-size=16g \
        -e TT_VISIBLE_DEVICES=$chip \
        registry.company.com/tt-forge-compiletron:v1.0 \
        python3 /app/scripts/docker/forge_worker.py production_job
done
```

### Monitoring

```bash
# Check running containers
docker ps | grep forge

# View logs
docker logs forge_full_chip_0

# Check compilation time
docker logs forge_full_chip_0 | grep "Compilation completed"
```

## Success Metrics

**Working as of:** 2026-03-26

- ✅ **4/4 chips** compile successfully in parallel
- ✅ **~6 seconds** compilation time per chip
- ✅ **No topology errors** (STRICT mesh descriptors working)
- ✅ **No symbol mismatches** (matched PyTorch versions)
- ✅ **All dependencies resolved** (7 iterative fixes)

**Reliability:** Production-ready for parallel workloads

## See Also

- `Dockerfile` - Full build configuration
- `Dockerfile.minimal.deprecated` - Why minimal approach failed
- `scripts/docker/run_parallel_4chip.sh` - 4-way test script
- `README.md` - Quick start guide
