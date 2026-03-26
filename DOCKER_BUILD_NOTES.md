# Docker Full-Build Implementation Notes

## Achievement

**Successfully implemented 4-way parallel Docker compilation** by building tt-metal and tt-forge-fe from source.

**Result:** 4/4 chips compile independently in ~6 seconds each.

## Why Full Source Build?

### Problem with Pre-built Images

Attempted to use `ghcr.io/tenstorrent/tt-forge-slim` pre-built images but encountered critical bugs:

**Topology Discovery Bug:**
```
ERROR: Physical chip id 1 not found in control plane chip mapping
UMD logs: n_log=1, n_phys=2
```

**Symptoms:**
- Chips 1 & 3: Immediate control plane crash
- Chips 0 & 2: Hang at "Starting topology discovery"
- TT_VISIBLE_DEVICES isolation not working

**Root Cause:**
- Forge-bundled tt-metal uses RELAXED mesh graph descriptor policy
- Incompatible with single-chip isolation via TT_VISIBLE_DEVICES
- Requires STRICT policy from standalone tt-metal build

### Solution

Build tt-metal and tt-forge-fe from source using:
1. Standalone tt-metal (not forge-bundled)
2. STRICT policy mesh graph descriptors
3. Matched library versions (no ABI conflicts)

## Build Iterations (10 Total)

### Iteration 1-5: Dependency Discovery
1. **libTTMLIRCompiler.so missing** → Added LD_LIBRARY_PATH
2. **libncurses.so.6 missing** → Added libncurses6
3. **libmpi.so.40 missing** → Added libopenmpi3t64
4. **MPI tools missing** → Added openmpi-bin
5. **loguru module missing** → Fixed Python environment

### Iteration 6-8: Python Environment
6. **PyTorch symbol mismatch** → Use toolchain venv pip (not system pip)
7. **System Python vs venv** → Add venv/bin to PATH
8. **TVM import error** → Reinstall tt_tvm with runtime paths

### Iteration 9-10: Final Fixes
9. **Mesh descriptor missing** → Copy mesh_graph_descriptors/ to image
10. **✅ SUCCESS** → 4/4 chips working!

## Dependencies Required

### System Libraries
```dockerfile
RUN apt-get install -y \
    libncurses6 \          # Terminal library for forge
    libopenmpi3t64 \       # MPI runtime library
    openmpi-bin \          # MPI tools and help files
    libhwloc15 \           # Hardware locality
    libnuma1 \             # NUMA support
    libboost-* \           # Boost libraries
    libyaml-cpp0.8         # YAML parsing
```

### Environment Variables
```dockerfile
ENV PATH=/opt/ttforge-toolchain/venv/bin:...  # Use venv Python
ENV LD_LIBRARY_PATH=/tt-forge-fe/third_party/tt-mlir/build/install/lib:...
ENV PYTHONPATH=/tt-metal:/tt-forge-fe/forge:...
```

### Application Files
```dockerfile
COPY mesh_graph_descriptors/ ./mesh_graph_descriptors/  # STRICT policy!
COPY lib/ scripts/ tests/ docs/ ...
```

## Critical Fixes

### 1. PyTorch Version Conflict

**Problem:** Runtime installed different PyTorch than build used
```
ImportError: undefined symbol: _ZNK3c106SymInt6sym_neERKS0_
```

**Solution:** Use toolchain venv pip instead of system pip
```dockerfile
# Wrong:
RUN pip3 install --break-system-packages -r requirements.txt

# Correct:
RUN /opt/ttforge-toolchain/venv/bin/pip install -r requirements.txt
```

### 2. TVM Module Path

**Problem:** tt_tvm installed in editable mode with builder path
```
installed: tt_tvm 0.14.0+dev /build/tt-forge-fe/third_party/tvm/python
runtime: /tt-forge-fe/third_party/tvm/python
```

**Solution:** Reinstall in runtime with correct path
```dockerfile
RUN /opt/ttforge-toolchain/venv/bin/pip install -e /tt-forge-fe/third_party/tvm/python
```

### 3. PATH Order

**Problem:** `python3` resolved to system Python, not venv
```
ModuleNotFoundError: No module named 'loguru'
# loguru in venv, but using system Python
```

**Solution:** Put venv/bin first in PATH
```dockerfile
ENV PATH=/opt/ttforge-toolchain/venv/bin:/opt/ttforge-toolchain/bin:${PATH}
```

## Image Specifications

**Final Image:**
- **Size:** 21GB content (71.9GB with layers)
- **Build Time:** 2-3 hours (first build)
- **Rebuild Time:** 5-10 minutes (cached builder layers)
- **Base:** Ubuntu 24.04
- **Python:** 3.12 (in toolchain venv)
- **PyTorch:** 2.7.0+cpu (from toolchain)

**What's Included:**
- tt-metal compiled from source
- tt-forge-fe compiled from source
- All dependencies (TensorFlow, JAX, PyTorch, ONNX, etc.)
- Mesh graph descriptors (STRICT policy)
- Compiletron application and model library

## Testing Results

### Single Chip Test
```bash
docker run ... -e TT_VISIBLE_DEVICES=0 ... tt-forge-compiletron:full
# ✓ Compilation completed in 5.87s
# ✓ Worker finished successfully!
```

### 4-Way Parallel Test
```bash
./scripts/docker/run_parallel_4chip.sh
# Chip 0: 5.87s ✓
# Chip 1: 5.74s ✓
# Chip 2: 6.21s ✓
# Chip 3: 6.12s ✓
# 🎉 ALL 4 CHIPS COMPILED SUCCESSFULLY!
```

## Future Migration Plan

### torch_plugin_tt (When Stable)

**Goal:** Smaller images using PyPI packages instead of source build

**Approach:**
```python
import torch
import torch_plugin_tt  # Registers "tt" backend

model = torch.nn.Linear(128, 128)
compiled = torch.compile(model, backend="tt")
output = compiled(inputs)
```

**Benefits:**
- Image size: 21GB → 10-15GB
- Build time: 2-3 hours → 15 minutes
- Maintenance: Track versions vs track source commits

**Blockers (as of 2026-03-26):**
- Circular import bug in torch_plugin_tt
```python
# torch_plugin_tt imports torch_xla
# torch_xla tries to register plugins
# imports torch_plugin_tt.TTPlugin
# AttributeError: partially initialized module
```
- Breaking API changes in dev releases (March 23 & 26)
- `forge` module removed, replaced with plugin architecture
- Not yet stable for production use

**Timeline:** Monitor forge releases for stable plugin API

## Lessons Learned

1. **Pre-built images aren't always portable** - Library bugs can block critical features
2. **Source builds ensure compatibility** - Control over build configuration matters
3. **Docker layer caching is powerful** - 2-3 hour builds become 5-minute rebuilds
4. **Iterative debugging works** - 10 iterations to identify all missing dependencies
5. **Environment isolation is complex** - venv vs system Python, builder vs runtime paths
6. **Document as you go** - This file captures 10+ hours of debugging insights

## Production Recommendations

1. **Build once, deploy everywhere** - Push to registry, pull on workers
2. **Use cached layers** - Don't delete builder cache between runs
3. **Monitor image size** - 21GB is manageable but plan for storage
4. **Test thoroughly** - Verify all chips work before production deploy
5. **Plan migration** - Be ready to switch to plugin API when stable

## Maintainer Notes

**Last Updated:** 2026-03-26
**Status:** Production-ready for 4-way parallel compilation
**Docker Image:** tt-forge-compiletron:full
**Verified On:** 4x P300C Blackhole chips

**Contact:** See README.md for issues/questions
