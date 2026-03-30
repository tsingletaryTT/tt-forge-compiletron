# Container Deployment Guide

Complete guide to deploying and running TT-Forge Compiletron as a containerized, dependable re-run system.

## 🎯 Goals

✅ **Portable** - Run anywhere with Docker
✅ **Reproducible** - Same environment every time
✅ **Automated** - Easy scheduling and re-running
✅ **Isolated** - No conflicts with host system
✅ **Persistent** - Cache and results survive container restarts

## 🚀 Quick Start (3 Steps)

### 1. Build the Container

```bash
cd ~/code/tt-forge-compiletron
make build
# Or: ./docker-build.sh
```

**Build time:** ~5-10 minutes (first build)
**Image size:** ~2GB

### 2. Run Tests

```bash
make test
# Or: ./docker-run.sh test
```

**Output:**
```
✅ All tests passed!
============================== 29 passed in 0.04s ==============================
```

### 3. Detect Hardware

```bash
make detect
# Or: ./docker-run.sh detect
```

**Output:**
```
✓ Detected 4 Tenstorrent chip(s)
  Board type: P300C
  Architecture: blackhole
  ...
```

## 📦 What's Included

### Container Components

```
tt-forge-compiletron:latest
├── Ubuntu 24.04 base
├── Python 3.12
├── All dependencies (PyTorch, torchvision, etc.)
├── Compiletron application
├── Test suite (29 tests)
└── Documentation
```

### Volume Mounts

```
Host                          → Container           Purpose
~/tt-metal/                   → /tt-metal/         TT-Metal (read-only)
~/tt-forge-fe/                → /tt-forge-fe/      Forge (read-only)
compiletron-cache (volume)    → /cache/            Model cache (persistent)
compiletron-results (volume)  → /results/          Results (persistent)
```

### Device Access

```
Host                    → Container
/dev/tenstorrent        → /dev/tenstorrent
```

## 🔧 Usage Patterns

### Pattern 1: One-Shot Commands

Run a command and exit:

```bash
# Detect hardware
./docker-run.sh detect

# Show model statistics
./docker-run.sh models stats

# List models
./docker-run.sh models list --family resnet
```

**Use case:** Quick queries, CI/CD, automation scripts

### Pattern 2: Compilation Runs

Run model compilations:

```bash
# Quick test (5 fastest models)
make compile-quick

# Parallel compilation (50 models on all chips)
make compile-parallel

# Custom run
./docker-run.sh compile --count 30 --parallel
```

**Use case:** Model validation, performance testing

### Pattern 3: Interactive Development

Enter container for exploration:

```bash
# Start interactive shell
make shell

# Inside container:
python3 compiletron.py detect
python3 compiletron.py models families
source /tt-forge-fe/env/activate
python3 lib/worker.py  # Run worker directly
```

**Use case:** Debugging, experimentation, development

### Pattern 4: Long-Running Service

Keep container running:

```bash
# Start as service
docker-compose up -d

# Execute commands
docker-compose exec compiletron python3 compiletron.py detect

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

**Use case:** Server deployments, scheduled jobs

### Pattern 5: Scheduled Automation

Cron-based execution:

```bash
# Add to crontab
crontab -e

# Run every night at 2 AM
0 2 * * * cd /path/to/tt-forge-compiletron && ./docker-run.sh compile --count 50 --parallel >> /var/log/compiletron.log 2>&1

# Run weekly stress test
0 3 * * 0 cd /path/to/tt-forge-compiletron && ./docker-run.sh compile --stress >> /var/log/compiletron-stress.log 2>&1
```

**Use case:** Continuous validation, regression testing

## 🎮 Command Reference

### Makefile Targets

```bash
make help               # Show all targets
make build              # Build Docker image
make test               # Run test suite
make detect             # Detect hardware
make shell              # Interactive shell
make models             # List all models
make stats              # Model statistics
make quick              # Quick test models
make compile-quick      # Compile 5 fastest models
make compile-parallel   # Compile 50 models in parallel
make clean              # Remove containers and volumes
```

### Direct Docker Commands

```bash
# Build
./docker-build.sh

# Run commands
./docker-run.sh <command> [args]

# Examples
./docker-run.sh detect
./docker-run.sh models stats
./docker-run.sh models list --family efficientnet
./docker-run.sh compile --quick
./docker-run.sh compile --parallel --count 50
./docker-run.sh test
./docker-run.sh shell
```

### Docker Compose

```bash
# Start
docker-compose up -d

# Execute
docker-compose exec compiletron <command>

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔄 Dependable Re-Run System

### Feature: Idempotent Execution

Running the same command multiple times is safe:

```bash
# First run: downloads models, compiles
./docker-run.sh compile --count 10

# Second run: uses cached models, skips already-compiled
./docker-run.sh compile --count 10
```

**Cache is preserved** in persistent volumes.

### Feature: Resume on Failure

If compilation fails partway through:

```bash
# Run with 50 models
./docker-run.sh compile --count 50

# Fails after model 23 → results saved

# Resume from where it stopped
./docker-run.sh compile --count 50 --skip-tested
```

**Results are persistent** across container restarts.

### Feature: Scheduled Validation

Continuous validation system:

```bash
# Create systemd timer (Linux)
sudo systemctl edit --force --full compiletron.timer

[Unit]
Description=TT-Forge Compiletron Nightly Run
Requires=compiletron.service

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target

# Create service
sudo systemctl edit --force --full compiletron.service

[Unit]
Description=TT-Forge Compiletron Compilation Run

[Service]
Type=oneshot
User=ttuser
WorkingDirectory=/home/ttuser/code/tt-forge-compiletron
ExecStart=/home/ttuser/code/tt-forge-compiletron/docker-run.sh compile --count 50 --parallel
StandardOutput=append:/var/log/compiletron.log
StandardError=append:/var/log/compiletron.log

# Enable and start
sudo systemctl enable compiletron.timer
sudo systemctl start compiletron.timer

# Check status
sudo systemctl status compiletron.timer
journalctl -u compiletron.service -f
```

### Feature: Health Checks

Automated validation:

```bash
#!/bin/bash
# health-check.sh

set -e

# Test container health
./docker-run.sh test || exit 1

# Test hardware detection
./docker-run.sh detect || exit 1

# Test model stats
./docker-run.sh models stats || exit 1

echo "✅ All health checks passed"
```

## 📊 Monitoring and Logging

### View Logs

```bash
# Container logs (if running via docker-compose)
docker-compose logs -f

# Specific service
docker logs -f tt-forge-compiletron

# Results directory
docker run --rm \
    -v compiletron-results:/results \
    ubuntu ls -la /results/
```

### Extract Results

```bash
# Copy results from container volume to host
docker run --rm \
    -v compiletron-results:/data \
    -v $(pwd):/backup \
    ubuntu cp /data/results.csv /backup/

# Or mount results directory directly
docker run --rm \
    -v compiletron-results:/results \
    -v $(pwd)/output:/output \
    ubuntu cp -r /results/* /output/
```

### Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Check compilation progress

CONTAINER="tt-forge-compiletron"

while true; do
    echo "=== Compiletron Status ==="
    echo "Container: $(docker ps -f name=$CONTAINER --format '{{.Status}}')"

    # Check results
    COMPLETED=$(docker run --rm -v compiletron-results:/results ubuntu bash -c "wc -l < /results/results.csv" 2>/dev/null || echo "0")
    echo "Models compiled: $COMPLETED"

    # Check cache size
    CACHE_SIZE=$(docker run --rm -v compiletron-cache:/cache ubuntu bash -c "du -sh /cache 2>/dev/null" || echo "unknown")
    echo "Cache size: $CACHE_SIZE"

    echo ""
    sleep 60
done
```

## 🔒 Security Considerations

### Read-Only Mounts

tt-metal and tt-forge-fe are mounted read-only:

```bash
-v ~/tt-metal:/tt-metal:ro
-v ~/tt-forge-fe:/tt-forge-fe:ro
```

### Device Permissions

Tenstorrent device requires appropriate permissions:

```bash
# Check permissions
ls -l /dev/tenstorrent

# If needed, adjust
sudo chmod 666 /dev/tenstorrent
```

### User Mapping

Container runs as root by default. For better security:

```dockerfile
# Add to Dockerfile
RUN useradd -m -u 1000 compiletron
USER compiletron
```

Or use docker user flag:

```bash
docker run --user $(id -u):$(id -g) ...
```

## 🚨 Troubleshooting

### Problem: Image Won't Build

**Symptoms:** Docker build fails

**Solutions:**
```bash
# Clear build cache
docker builder prune -a

# Rebuild from scratch
docker build --no-cache -t tt-forge-compiletron:latest .

# Check disk space
df -h
```

### Problem: Hardware Not Detected

**Symptoms:** Container shows "No hardware detected"

**Solutions:**
```bash
# Check device exists on host
ls -l /dev/tenstorrent

# Check device is passed to container
docker run --device=/dev/tenstorrent ...

# Test on host first
python3 compiletron.py detect
```

### Problem: Forge Not Found

**Symptoms:** "tt-forge-fe not found"

**Solutions:**
```bash
# Check path on host
ls -la ~/tt-forge-fe/env/activate

# Use absolute path in mount
docker run -v /home/ttuser/tt-forge-fe:/tt-forge-fe:ro ...

# Set environment variable
export FORGE_HOME=/path/to/tt-forge-fe
./docker-run.sh compile --quick
```

### Problem: Out of Disk Space

**Symptoms:** Container or build fails due to space

**Solutions:**
```bash
# Check Docker disk usage
docker system df

# Clean up
docker system prune -a --volumes

# Check volume sizes
docker volume ls
docker volume inspect compiletron-cache
```

### Problem: Slow Compilation

**Symptoms:** Compilation takes longer than expected

**Solutions:**
```bash
# Allocate more resources
docker run --cpus=16 --memory=64g ...

# Use parallel execution
./docker-run.sh compile --parallel

# Check if cache is working
docker run --rm -v compiletron-cache:/cache ubuntu ls -lh /cache/
```

## 📈 Performance Optimization

### Build Optimization

```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build -t tt-forge-compiletron:latest .

# Multi-stage build (already optimized in Dockerfile)

# Layer caching
# Requirements copied before app code for better caching
```

### Runtime Optimization

```bash
# Allocate more resources
docker run \
    --cpus=16 \
    --memory=64g \
    --shm-size=16g \
    ...

# Use tmpfs for temporary files
docker run \
    --tmpfs /tmp:rw,size=8g \
    ...

# Host networking (if applicable)
docker run --network=host ...
```

### Volume Performance

```bash
# Use local volume driver
docker volume create \
    --driver local \
    --opt type=none \
    --opt device=/fast/ssd/path \
    --opt o=bind \
    compiletron-cache-fast

# Or use NFS for shared access
docker volume create \
    --driver local \
    --opt type=nfs \
    --opt o=addr=nfs.server,rw \
    --opt device=:/path/to/share \
    compiletron-results-shared
```

## 🌐 Multi-Host Deployment

### Kubernetes Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: tt-forge-compiletron
spec:
  containers:
  - name: compiletron
    image: tt-forge-compiletron:latest
    command: ["python3", "compiletron.py", "compile", "--parallel"]
    volumeMounts:
    - name: tt-metal
      mountPath: /tt-metal
      readOnly: true
    - name: tt-forge-fe
      mountPath: /tt-forge-fe
      readOnly: true
    - name: cache
      mountPath: /cache
    resources:
      limits:
        tenstorrent.com/device: 1
  volumes:
  - name: tt-metal
    hostPath:
      path: /opt/tt-metal
  - name: tt-forge-fe
    hostPath:
      path: /opt/tt-forge-fe
  - name: cache
    persistentVolumeClaim:
      claimName: compiletron-cache
```

## 📚 See Also

- [Container Usage Guide](docs/CONTAINER_USAGE.md) - Detailed usage
- [Main README](README.md) - Project overview
- [Test Documentation](tests/README.md) - Test suite info
- [Hardware Detection Fix](HARDWARE_DETECTION_FIX.md) - Recent fixes
