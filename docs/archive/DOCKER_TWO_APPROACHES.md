# Docker: Two Approaches

TT-Forge Compiletron supports two Docker strategies, each with different tradeoffs.

## Approach 1: Reference Image (Fast, Lightweight)

**File:** `Dockerfile`
**Build Script:** `./docker-build.sh` or `make build`
**Image Size:** ~16GB
**Build Time:** ~6 minutes

### What It Includes
- ✅ All 50+ Python dependencies (TensorFlow, JAX, PyTorch, etc.)
- ✅ Compiletron application
- ✅ Test suite (29 tests)
- ✅ Model library (101 models)

### What You Need to Provide
- 📦 tt-metal (mount from host: `-v ~/tt-metal:/tt-metal:ro`)
- 📦 tt-forge-fe (mount from host: `-v ~/tt-forge-fe:/tt-forge-fe:ro`)
- 🔌 Hardware (mount device: `--device=/dev/tenstorrent`)

### Use Cases
✅ **Testing** - Run 29 tests without hardware
✅ **Model queries** - Browse 101 models
✅ **CI/CD** - Validate changes automatically
✅ **Quick iteration** - Fast builds for development

### Build Command
```bash
./docker-build.sh
# Or: make build
```

### Run Examples
```bash
# Test (no hardware needed)
make test

# Model statistics
make stats

# Compilation (requires mounts + hardware)
./docker-run.sh compile --quick
```

---

## Approach 2: Full Self-Contained Image (Complete, Heavy)

**File:** `Dockerfile.full-build`
**Build Script:** `./docker-build-full.sh`
**Image Size:** ~30GB
**Build Time:** 2-3 hours

### What It Includes
- ✅ All Python dependencies
- ✅ Compiletron application
- ✅ **tt-metal built from source**
- ✅ **tt-forge-fe built from source**
- ✅ All compiled C++ extensions

### What You Need to Provide
- 🔌 Hardware only (mount device: `--device=/dev/tenstorrent`)

### Use Cases
✅ **Complete portability** - No host dependencies
✅ **Production deployment** - Build once, run everywhere
✅ **Reproducible builds** - Pin exact commits
✅ **Air-gapped environments** - No external dependencies at runtime

### Build Command
```bash
./docker-build-full.sh

# With specific commits
./docker-build-full.sh \
    --tt-metal-commit e867533 \
    --tt-forge-commit 22be241 \
    --tag production-v1.0
```

### Run Examples
```bash
# Test (no hardware needed)
docker run --rm tt-forge-compiletron:full-latest test

# Detect hardware
docker run --rm --device=/dev/tenstorrent \
    tt-forge-compiletron:full-latest detect

# Compile models (no host mounts needed!)
docker run --rm --device=/dev/tenstorrent \
    -v compiletron-cache:/cache \
    -v compiletron-results:/results \
    tt-forge-compiletron:full-latest compile --quick
```

---

## Comparison Table

| Feature | Reference | Full Build |
|---------|-----------|------------|
| **Build Time** | 6 min | 2-3 hours |
| **Image Size** | 16GB | 30GB |
| **Build Requirements** | 8GB RAM | 32GB+ RAM |
| **Disk Space** | 20GB | 50GB+ |
| **Python Deps** | ✅ Included | ✅ Included |
| **tt-metal** | ❌ Mount from host | ✅ Built-in |
| **tt-forge-fe** | ❌ Mount from host | ✅ Built-in |
| **C++ Extensions** | ⚠️ From host | ✅ Built-in |
| **Compilation** | ⚠️ With mounts | ✅ Self-contained |
| **Testing** | ✅ Works | ✅ Works |
| **CI/CD** | ✅ Practical | ⚠️ Expensive |
| **Development** | ✅ Fast iteration | ❌ Slow rebuilds |
| **Production** | ⚠️ Needs host setup | ✅ True portability |

---

## When to Use Each

### Use Reference Image When:
- 🚀 You want fast builds (6 min vs 2-3 hours)
- 💻 Developing locally with tt-metal/forge already built
- 🧪 Running tests in CI/CD
- 📊 Querying model library
- 🔄 Iterating quickly on compiletron code

### Use Full Build When:
- 🌍 Deploying to production servers
- 📦 Need complete reproducibility
- 🔒 Working in air-gapped environments
- 🎯 Want true "docker run and forget" experience
- 📌 Need specific pinned commits of tt-metal/forge

---

## Build Instructions

### Reference Image
```bash
# Quick build
make build

# Or with script
./docker-build.sh

# Test it
make test
make stats
```

### Full Self-Contained Image
```bash
# Default build (uses latest main branches)
./docker-build-full.sh

# Production build with specific commits
./docker-build-full.sh \
    --tt-metal-commit e867533 \
    --tt-forge-commit 22be241 \
    --tag prod-2026-03-24

# Force rebuild without cache
./docker-build-full.sh --no-cache

# ⚠️ WARNING: Plan for 2-3 hours and 32GB+ RAM!
```

---

## Resource Requirements

### Reference Image Build
```
RAM: 8GB
CPU: 4 cores
Disk: 20GB
Time: 6 minutes
Network: ~3GB downloads
```

### Full Build
```
RAM: 32GB+ (tt-metal build is memory-intensive)
CPU: 8+ cores (parallel builds)
Disk: 50GB+ (build artifacts + final image)
Time: 2-3 hours
Network: ~10GB downloads
```

---

## Troubleshooting

### Reference Image Issues

**"Forge not found"**
```bash
# Verify tt-forge-fe is mounted
docker run --rm \
    -v ~/tt-forge-fe:/tt-forge-fe:ro \
    tt-forge-compiletron:latest \
    bash -c "ls -la /tt-forge-fe"
```

**"Cannot import forge"**
- C++ extensions (`forge._C`) must be built on host
- Mount the compiled forge from host
- Or use full build image

### Full Build Issues

**"Out of memory during build"**
- Increase Docker memory limit to 32GB+
- Close other applications
- Use machine with more RAM

**"Build taking forever"**
- tt-metal build: ~45-60 minutes (normal)
- tt-forge build: ~45-60 minutes (normal)
- Total: 2-3 hours is expected
- Use `--no-cache` only when necessary

**"Disk space issues"**
- Need 50GB+ during build
- Final image is ~30GB
- Clean up: `docker system prune -a`

---

## Performance Considerations

### Reference Image
- ✅ Fast startup (seconds)
- ✅ Native host performance
- ✅ Quick rebuilds
- ⚠️ Requires host setup

### Full Build
- ✅ True portability
- ✅ No host dependencies
- ⚠️ Slower startup (container overhead)
- ⚠️ Slow rebuilds (2-3 hours)

---

## Recommendation

**For most users:** Start with **Reference Image**
- Fast to build
- Easy to iterate
- Good for development and testing

**For production:** Use **Full Build**
- Build once, deploy everywhere
- No host dependencies
- Reproducible and portable

**Best of both worlds:**
1. Develop with reference image
2. Deploy with full build
3. Use CI/CD with reference image (fast tests)
4. Build full image periodically for releases

---

## Next Steps

### To Get Started with Reference Image:
```bash
make build
make test
./docker-run.sh models stats
```

### To Build Full Self-Contained Image:
```bash
# ⚠️ Allocate 2-3 hours!
./docker-build-full.sh

# When done (2-3 hours later):
docker run --rm tt-forge-compiletron:full-latest test
```

### To Use in Production:
```bash
# Build with specific versions
./docker-build-full.sh \
    --tt-metal-commit <stable-commit> \
    --tt-forge-commit <stable-commit> \
    --tag v1.0.0

# Deploy
docker tag tt-forge-compiletron:v1.0.0 registry.company.com/tt-forge:v1.0.0
docker push registry.company.com/tt-forge:v1.0.0
```

---

**Questions?** See [DOCKER_REFERENCE.md](DOCKER_REFERENCE.md) for more details.
