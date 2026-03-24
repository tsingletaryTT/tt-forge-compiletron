# Container Usage Guide

Complete guide to running TT-Forge Compiletron in Docker.

## Quick Start

### 1. Build the Image

```bash
# Using Makefile (recommended)
make build

# Or directly
./docker-build.sh

# Or with docker command
docker build -t tt-forge-compiletron:latest .
```

### 2. Run Basic Commands

```bash
# Detect hardware
make detect
# Or: ./docker-run.sh detect

# Show model statistics
make stats
# Or: ./docker-run.sh models stats

# Run test suite
make test
# Or: ./docker-run.sh test
```

### 3. Compile Models

```bash
# Quick test (5 fastest models)
make compile-quick

# Parallel compilation (50 models, all chips)
make compile-parallel

# Custom compilation
./docker-run.sh compile --count 20 --parallel
```

## Architecture

### Container Structure

```
/app/                           # Application root
├── compiletron.py             # Main CLI
├── lib/                       # Core modules
├── scripts/                   # Helper scripts
├── docs/                      # Documentation
└── tests/                     # Test suite

/tt-metal/                     # Mounted from host (read-only)
/tt-forge-fe/                  # Mounted from host (read-only)
/cache/                        # Persistent volume (PyTorch cache)
/results/                      # Persistent volume (compilation results)
/models/                       # Optional custom models
```

### Volume Mounts

**Required:**
- `/tt-metal` ← `~/tt-metal` (read-only)
- `/tt-forge-fe` ← `~/tt-forge-fe` (read-only)

**Persistent:**
- `/cache` ← Docker volume `compiletron-cache`
- `/results` ← Docker volume `compiletron-results`

**Optional:**
- `/models` ← `./models` or custom directory

### Device Access

Container needs access to `/dev/tenstorrent` for hardware detection and compilation.

## Usage Patterns

### Pattern 1: One-Shot Commands

Run a single command and exit:

```bash
# Detect hardware
./docker-run.sh detect

# List models by family
./docker-run.sh models list --family resnet

# Estimate compilation time
./docker-run.sh models estimate --count 50 --chips 4

# Run quick test
./docker-run.sh compile --quick
```

### Pattern 2: Interactive Shell

Start an interactive session:

```bash
# Enter shell
./docker-run.sh shell

# Inside container:
python3 compiletron.py detect
python3 compiletron.py models stats
source /tt-forge-fe/env/activate
python3 compiletron.py run --quick
```

### Pattern 3: Docker Compose

Long-running service:

```bash
# Start container
docker-compose up -d

# Execute commands
docker-compose exec compiletron python3 compiletron.py detect
docker-compose exec compiletron python3 compiletron.py models stats

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Pattern 4: Scheduled Runs (Cron)

Automated compilation runs:

```bash
# Add to crontab
0 2 * * * cd /path/to/tt-forge-compiletron && ./docker-run.sh compile --count 50 --parallel >> /var/log/compiletron.log 2>&1
```

## Commands Reference

### Hardware Detection

```bash
# Detect all hardware
./docker-run.sh detect

# Check specific details
./docker-run.sh shell
> python3 compiletron.py detect
```

### Model Operations

```bash
# List all models
./docker-run.sh models list

# Filter by family
./docker-run.sh models list --family efficientnet

# Filter by complexity
./docker-run.sh models list --complexity low

# Show statistics
./docker-run.sh models stats

# Show families
./docker-run.sh models families

# Get model info
./docker-run.sh models info ResNet-50

# Quick test models
./docker-run.sh models quick

# Stress test models
./docker-run.sh models stress

# Estimate time
./docker-run.sh models estimate --count 50 --chips 4
```

### Compilation

```bash
# Quick test (5 fastest models)
./docker-run.sh compile --quick

# Stress test (5 slowest models)
./docker-run.sh compile --stress

# Specific count
./docker-run.sh compile --count 20

# Parallel execution
./docker-run.sh compile --parallel --count 50

# Specific family
./docker-run.sh compile --family resnet

# With custom settings
./docker-run.sh compile \
    --count 30 \
    --complexity low \
    --parallel
```

### Testing

```bash
# Run all tests
./docker-run.sh test

# Run specific test file
docker run --rm tt-forge-compiletron:latest \
    python3 -m pytest tests/test_hardware.py -v

# Run with coverage
docker run --rm tt-forge-compiletron:latest \
    python3 -m pytest --cov=lib tests/
```

## Environment Variables

### Host-Side (before docker run)

```bash
# Override default paths
export TT_METAL_HOME=/custom/path/to/tt-metal
export FORGE_HOME=/custom/path/to/tt-forge-fe

# Then run
./docker-run.sh detect
```

### Container-Side (inside Docker)

```bash
# These are set automatically by docker-run.sh
TT_METAL_HOME=/tt-metal
FORGE_HOME=/tt-forge-fe
CACHE_DIR=/cache
RESULTS_DIR=/results
PYTHONPATH=/tt-metal:${PYTHONPATH}
```

### Custom Variables

```bash
# Pass custom variables
docker run --rm \
    --device=/dev/tenstorrent \
    -v ~/tt-metal:/tt-metal:ro \
    -v ~/tt-forge-fe:/tt-forge-fe:ro \
    -e MY_CUSTOM_VAR=value \
    tt-forge-compiletron:latest detect
```

## Volume Management

### Inspect Volumes

```bash
# List volumes
docker volume ls | grep compiletron

# Inspect cache volume
docker volume inspect compiletron-cache

# Inspect results volume
docker volume inspect compiletron-results
```

### Backup Volumes

```bash
# Backup cache
docker run --rm \
    -v compiletron-cache:/data \
    -v $(pwd):/backup \
    ubuntu tar czf /backup/cache-backup.tar.gz /data

# Backup results
docker run --rm \
    -v compiletron-results:/data \
    -v $(pwd):/backup \
    ubuntu tar czf /backup/results-backup.tar.gz /data
```

### Restore Volumes

```bash
# Restore cache
docker run --rm \
    -v compiletron-cache:/data \
    -v $(pwd):/backup \
    ubuntu tar xzf /backup/cache-backup.tar.gz -C /

# Restore results
docker run --rm \
    -v compiletron-results:/data \
    -v $(pwd):/backup \
    ubuntu tar xzf /backup/results-backup.tar.gz -C /
```

### Clean Volumes

```bash
# Remove all compiletron volumes
docker volume rm compiletron-cache compiletron-results

# Or with docker-compose
docker-compose down -v
```

## Advanced Usage

### Multi-Chip Parallel Execution

```bash
# Run on all detected chips
./docker-run.sh compile --parallel --count 100

# The container will:
# 1. Detect number of chips (e.g., 4)
# 2. Calculate round-robin distribution (25 models per chip)
# 3. Run 4 worker processes in parallel
# 4. Aggregate results
```

### Custom Model Directory

```bash
# Mount custom models
docker run --rm \
    --device=/dev/tenstorrent \
    -v ~/tt-metal:/tt-metal:ro \
    -v ~/tt-forge-fe:/tt-forge-fe:ro \
    -v ~/my-models:/models:ro \
    -v compiletron-cache:/cache \
    tt-forge-compiletron:latest compile --models-dir /models
```

### Debug Mode

```bash
# Run with debug output
docker run --rm -it \
    --device=/dev/tenstorrent \
    -v ~/tt-metal:/tt-metal:ro \
    -v ~/tt-forge-fe:/tt-forge-fe:ro \
    -e PYTHONUNBUFFERED=1 \
    -e DEBUG=1 \
    tt-forge-compiletron:latest shell

# Inside:
python3 -u compiletron.py detect
```

### Resource Limits

```bash
# Limit CPU and memory
docker run --rm \
    --device=/dev/tenstorrent \
    --cpus=8 \
    --memory=32g \
    -v ~/tt-metal:/tt-metal:ro \
    -v ~/tt-forge-fe:/tt-forge-fe:ro \
    tt-forge-compiletron:latest compile --count 50
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: TT-Forge Compiletron CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t tt-forge-compiletron:ci .

      - name: Run tests
        run: docker run --rm tt-forge-compiletron:ci test

      - name: Run model stats
        run: docker run --rm tt-forge-compiletron:ci models stats
```

### GitLab CI Example

```yaml
test:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t tt-forge-compiletron:ci .
    - docker run --rm tt-forge-compiletron:ci test
    - docker run --rm tt-forge-compiletron:ci models stats
```

## Troubleshooting

### Image Build Fails

**Problem:** Docker build fails during dependency installation

**Solution:**
```bash
# Clear Docker cache
docker builder prune -a

# Rebuild without cache
docker build --no-cache -t tt-forge-compiletron:latest .
```

### Hardware Not Detected

**Problem:** Container can't see Tenstorrent devices

**Solution:**
```bash
# Check device exists on host
ls -l /dev/tenstorrent

# Ensure device is passed to container
docker run --device=/dev/tenstorrent ...

# Check permissions
sudo chmod 666 /dev/tenstorrent
```

### Volume Mount Issues

**Problem:** tt-metal or tt-forge-fe not found

**Solution:**
```bash
# Verify paths on host
ls -la ~/tt-metal
ls -la ~/tt-forge-fe

# Use absolute paths
docker run -v /home/user/tt-metal:/tt-metal:ro ...
```

### Compilation Fails

**Problem:** Models fail to compile in container

**Solution:**
```bash
# Check Forge environment
./docker-run.sh shell
> source /tt-forge-fe/env/activate
> python3 -c "import forge; print(forge.__version__)"

# Check hardware access
> python3 compiletron.py detect

# Try single model
> python3 compiletron.py run --count 1
```

## Performance Tuning

### Build Time Optimization

```bash
# Use BuildKit for parallel builds
DOCKER_BUILDKIT=1 docker build -t tt-forge-compiletron:latest .

# Multi-stage build (if needed)
# Already optimized in current Dockerfile
```

### Runtime Optimization

```bash
# Increase shared memory
docker run --shm-size=16g ...

# Use host network (if applicable)
docker run --network=host ...

# Disable DNS lookup
docker run --dns=8.8.8.8 ...
```

## See Also

- [Main README](../README.md)
- [Hardware Detection](MULTI_CHIP.md)
- [Model Library](MODEL_LIBRARY.md)
- [Test Documentation](../tests/README.md)
