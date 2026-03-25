# Docker Reference Implementation

This Docker image is a **reference implementation** showing how to build a complete TT-Forge compilation environment.

## What This Image Includes

✅ **Full Forge Dependencies** - All 50+ Python packages needed for compilation
✅ **PyTorch** - Required for model loading
✅ **TensorFlow** - Required by Forge's TVM integration
✅ **JAX** - Required by Forge
✅ **ONNX** - For model conversion
✅ **All Compiletron Tools** - Hardware detection, model library, worker

## Image Specifications

- **Base:** Ubuntu 24.04
- **Python:** 3.12
- **Size:** ~10GB (full stack)
- **Build Time:** 15-20 minutes

## What Users Still Need to Provide

Users must mount these from their host system:

### 1. TT-Metal
```bash
-v ~/tt-metal:/tt-metal:ro
```
Users build this themselves from https://github.com/tenstorrent/tt-metal

### 2. TT-Forge-FE
```bash
-v ~/tt-forge-fe:/tt-forge-fe:ro
```
Users build this themselves from https://github.com/tenstorrent/tt-forge-fe

### 3. Hardware Access
```bash
--device=/dev/tenstorrent
```
Requires Tenstorrent hardware with firmware installed

## Usage

### Build Image
```bash
make build
# Or: ./docker-build.sh
```

**Note:** First build takes 15-20 minutes and requires good internet connection for downloading dependencies.

### Run Compilation
```bash
./docker-run.sh compile --quick
```

This mounts:
- tt-metal from `~/tt-metal`
- tt-forge-fe from `~/tt-forge-fe`
- Tenstorrent hardware via `/dev/tenstorrent`

### Run Tests (No Hardware Needed)
```bash
./docker-run.sh test
```

Tests use mock data and don't require hardware or tt-metal/forge mounts.

## Why This Approach?

**Benefits:**
1. **Reproducible** - Same environment every time
2. **Reference** - Shows complete dependency stack
3. **Portable** - Runs anywhere Docker runs
4. **Testable** - Can run tests without hardware

**Tradeoffs:**
1. **Large Image** - 10GB vs 2GB minimal
2. **Build Time** - 15-20 min vs 2-3 min
3. **Not for CI** - GitHub Actions can't build full stack
4. **Still Needs Mounts** - Metal/Forge must be built separately

## Alternative: Minimal Image

For CI/testing without compilation, you can create a minimal image:

```dockerfile
# Only install test dependencies
RUN pip3 install pytest numpy pyfiglet

# Skip forge dependencies
# Skip tensorflow, jax, etc.
```

This creates a 2GB image in 3 minutes, perfect for automated testing.

## Comparison

| Feature | Full Image | Minimal Image |
|---------|------------|---------------|
| Size | 10GB | 2GB |
| Build Time | 15-20 min | 2-3 min |
| Dependencies | 50+ packages | 5 packages |
| Compilation | ✅ Yes | ❌ No |
| Testing | ✅ Yes | ✅ Yes |
| CI Suitable | ❌ No | ✅ Yes |
| Reference | ✅ Yes | ❌ No |

## Production Deployment

For production, consider:

1. **Build Once** - Build full image, push to registry
2. **Pull Everywhere** - Workers pull pre-built image
3. **Mount Volumes** - Each worker mounts tt-metal/forge
4. **Hardware Access** - Workers need Tenstorrent devices

Example:
```bash
# Build once (development machine)
docker build -t registry.company.com/tt-forge-compiletron:v1.0 .
docker push registry.company.com/tt-forge-compiletron:v1.0

# Run everywhere (production machines)
docker pull registry.company.com/tt-forge-compiletron:v1.0
docker run --device=/dev/tenstorrent \
  -v /opt/tt-metal:/tt-metal:ro \
  -v /opt/tt-forge-fe:/tt-forge-fe:ro \
  registry.company.com/tt-forge-compiletron:v1.0 \
  compile --quick
```

## Troubleshooting

**Build fails downloading packages:**
- Check internet connection
- May need to configure pip proxy
- Some packages are large (tensorflow ~500MB)

**Compilation fails in container:**
- Verify tt-metal mounted correctly
- Verify tt-forge-fe mounted correctly
- Check hardware access: `ls -la /dev/tenstorrent`

**Image too large:**
- This is expected for full reference implementation
- Consider minimal image for CI
- Use `.dockerignore` to exclude test data

## Success Criteria

A successful build should:
1. ✅ Complete without errors
2. ✅ Install all 50+ packages
3. ✅ Pass 29 tests
4. ✅ Import forge module successfully
5. ✅ Compile 5 quick test models

Verified working: **2026-03-24**
