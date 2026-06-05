# TT-Forge Installation Guide

Comprehensive guide to building and installing tt-forge-fe from source.

## Prerequisites

- **Python 3.12+** (required)
- **tt-metal** installed (`~/tt-metal`)
- **Build tools**: gcc, g++, cmake, git
- **Disk space**: ~20GB for build
- **Time**: 45-60 minutes

## Quick Installation

Using the setup script (recommended — pip wheel, no source build):

```bash
cd ~/code/tt-forge-compiletron
bash scripts/setup-venvs.sh --forge
```

This creates `~/tt-forge-venv` from the Tenstorrent pip index and writes
`~/tt-forge-fe/env/activate` so the harness can activate it automatically.

For a full source build instead:

```bash
cd ~
git clone https://github.com/tenstorrent/tt-forge-fe.git
cd tt-forge-fe
./build.sh          # 45–90 min
source env/activate
```

## Manual Installation

### Step 1: Clone Repository

```bash
cd ~
git clone https://github.com/tenstorrent/tt-forge-fe.git
cd tt-forge-fe
```

### Step 2: Build Forge

```bash
./build_forge.sh
```

This script will:
- Build TTMLIR toolchain
- Build Forge compiler
- Create Python virtual environment
- Install Python dependencies

**Build time**: 45-60 minutes on typical workstation

**Monitor progress**:
```bash
# In another terminal
tail -f ~/tt-forge-fe/build.log
```

### Step 3: Activate Environment

After build completes:

```bash
source ~/tt-forge-fe/env/activate
```

This sets up:
- `TTFORGE_TOOLCHAIN_DIR`
- `TTFORGE_VENV_DIR`
- `TTMLIR_TOOLCHAIN_DIR`
- `ARCH_NAME`

## Verify Installation

### Test Python Import

```python
python3 -c "import forge; print(f'Forge version: {forge.__version__}')"
```

### Check Environment

```bash
cd ~/code/tt-forge-compiletron
python3 compiletron.py setup check
```

Should show:
- ✓ Forge installed
- ✓ Forge environment activated
- ✓ All dependencies present

### Test Compilation

Compile a simple model:

```bash
cd ~/code/tt-forge-compiletron
source ~/tt-forge-fe/env/activate

python3 -c "
import torch
import torchvision.models as models
import forge

# Create simple model
model = models.resnet18(pretrained=False)
model.eval()

# Create input
sample_input = torch.randn(1, 3, 224, 224)

# Compile
print('Compiling ResNet-18...')
compiled_model = forge.compile(model, sample_inputs=[sample_input])
print('✓ Compilation successful!')
"
```

## Common Build Issues

### Issue: Python Version Mismatch

**Error**: `Python 3.12+ required`

**Fix**:
```bash
# Install Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev

# Use specific Python version
python3.12 -m venv ~/tt-forge-fe/env
```

### Issue: Missing Build Tools

**Error**: `gcc not found` or `cmake not found`

**Fix**:
```bash
sudo apt install build-essential cmake git
```

### Issue: Out of Disk Space

**Error**: `No space left on device`

**Fix**:
- Need at least 20GB free space
- Clean up old builds: `rm -rf ~/tt-forge-fe/build`
- Check space: `df -h ~`

### Issue: Build Timeout

**Error**: Build hangs or takes > 2 hours

**Fix**:
- Check CPU usage: `top`
- Ensure adequate RAM (recommend 16GB+)
- Try with fewer parallel jobs:
  ```bash
  export MAX_JOBS=4
  ./build_forge.sh
  ```

### Issue: Import Error After Build

**Error**: `ImportError: No module named 'forge'`

**Fix**:
1. Ensure environment is activated:
   ```bash
   source ~/tt-forge-fe/env/activate
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

3. Manually add to path if needed:
   ```python
   import sys
   sys.path.insert(0, '/home/yourusername/tt-forge-fe')
   import forge
   ```

## Build Components

The build process creates:

### 1. TTMLIR Toolchain (`/opt/ttmlir-toolchain`)
- MLIR compiler infrastructure
- Flatbuffer compiler
- Custom MLIR dialects

### 2. Forge Compiler (`~/tt-forge-fe/forge`)
- Python frontend
- Graph optimization
- MLIR lowering
- TT-Metal backend

### 3. Python Environment (`~/tt-forge-fe/env`)
- Virtual environment with dependencies
- PyTorch 2.0+
- TVM for ONNX/TF support
- Python bindings

## Environment Variables

After activation, these are set:

```bash
TTFORGE_TOOLCHAIN_DIR=/opt/ttforge-toolchain
TTFORGE_PYTHON_VERSION=python3.12
TTFORGE_VENV_DIR=/opt/ttforge-toolchain/venv
TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
TTMLIR_VENV_DIR=/opt/ttmlir-toolchain/venv
ARCH_NAME=blackhole  # or wormhole_b0, grayskull
```

## Updating Forge

To update to latest version:

```bash
cd ~/tt-forge-fe
git pull
./build_forge.sh  # Rebuild (45-60 min)
```

## Uninstalling

To completely remove Forge:

```bash
# Remove source
rm -rf ~/tt-forge-fe

# Remove toolchains
sudo rm -rf /opt/ttforge-toolchain
sudo rm -rf /opt/ttmlir-toolchain

# Remove environment variables (from ~/.bashrc or ~/.zshrc)
# Remove any lines with TTFORGE or TTMLIR
```

## Next Steps

After successful installation:

1. **Verify compiletron setup**:
   ```bash
   cd ~/code/tt-forge-compiletron
   ./setup.sh
   ```

2. **Test quick models**:
   ```bash
   python3 compiletron.py models quick
   ```

3. **Try compilation**:
   ```bash
   python3 compiletron.py run --quick
   ```

## Alternative: Pre-built Packages (if available)

Check if pre-built wheels are available:

```bash
# Check PyPI (may not be available yet)
pip search tt-forge

# Check GitHub releases
# https://github.com/tenstorrent/tt-forge-fe/releases
```

## Support

- **Documentation**: https://docs.tenstorrent.com/tt-forge-fe/
- **GitHub Issues**: https://github.com/tenstorrent/tt-forge-fe/issues
- **Discord**: Tenstorrent community server

## See Also

- [Multi-Chip Setup](MULTI_CHIP.md) - Multi-chip configuration
- [Model Library](MODEL_LIBRARY.md) - Available models
- [Main README](../README.md) - Compiletron usage
