#!/bin/bash
# TT-Forge Compiletron - Setup Script

set -e

echo "🎰 TT-Forge Compiletron Setup"
echo "================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.12+."
    exit 1
fi

echo "✓ Found Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "⚠️  Python 3.12+ recommended (you have $PYTHON_VERSION)"
fi

# Check tt-metal
echo ""
echo "🔍 Checking for tt-metal..."
TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tt-metal}"

if [ -d "$TT_METAL_HOME" ]; then
    echo "✓ Found tt-metal: $TT_METAL_HOME"
else
    echo "⚠️  tt-metal not found at $TT_METAL_HOME"
    echo "   Install from: https://github.com/tenstorrent/tt-metal"
fi

# Check tt-forge-fe
echo ""
echo "🔍 Checking for tt-forge-fe..."
FORGE_HOME="${FORGE_HOME:-$HOME/tt-forge-fe}"

if [ -d "$FORGE_HOME" ]; then
    echo "✓ Found tt-forge-fe: $FORGE_HOME"

    if [ -f "$FORGE_HOME/env/activate" ]; then
        echo "  ✓ Environment available: source $FORGE_HOME/env/activate"
    else
        echo "  ⚠️  Environment not found. May need to rebuild."
    fi
else
    echo "⚠️  tt-forge-fe not found"
    echo ""
    echo "   Install options:"
    echo "   1. Use compiletron CLI:"
    echo "      python3 $SCRIPT_DIR/compiletron.py setup install-forge"
    echo ""
    echo "   2. Manual installation:"
    echo "      cd ~"
    echo "      git clone https://github.com/tenstorrent/tt-forge-fe.git"
    echo "      cd tt-forge-fe"
    echo "      ./build_forge.sh  # Takes 45-60 minutes"
    echo ""
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "⚠️  requirements.txt not found"
else
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
    echo "✓ Dependencies installed"
fi

# Check hardware
echo ""
echo "🔍 Checking for Tenstorrent hardware..."
if command -v tt-smi &> /dev/null; then
    # Use Python to parse JSON properly
    DEVICE_COUNT=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/lib')
from hardware import detect_hardware
hw = detect_hardware()
print(hw.get('num_chips', 0))
" 2>/dev/null || echo "0")

    if [ "$DEVICE_COUNT" -gt 0 ]; then
        echo "✓ Found $DEVICE_COUNT Tenstorrent device(s)"

        # Show detailed info
        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/lib')
from hardware import detect_hardware, print_hardware_info
hw = detect_hardware()
if hw.get('num_chips', 0) > 0:
    print_hardware_info(hw)
" 2>/dev/null || true
    else
        echo "⚠️  No Tenstorrent devices detected"
        echo "   Run 'tt-smi' for detailed information"
    fi
else
    echo "⚠️  tt-smi not found (not installed or not in PATH)"
fi

# Test CLI
echo ""
echo "🧪 Testing CLI..."
if python3 "$SCRIPT_DIR/compiletron.py" --help > /dev/null 2>&1; then
    echo "✓ CLI working"
else
    echo "⚠️  CLI test failed"
fi

# Summary
echo ""
echo "================================"
echo "Setup Summary"
echo "================================"

# Check what's ready
PYTHON_OK=1
TTMETAL_OK=0
FORGE_OK=0
HW_OK=0

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
    PYTHON_OK=1
fi

if [ -d "$TT_METAL_HOME" ]; then
    TTMETAL_OK=1
fi

if [ -d "$FORGE_HOME" ] && [ -f "$FORGE_HOME/env/activate" ]; then
    FORGE_OK=1
fi

if [ "$DEVICE_COUNT" -gt 0 ]; then
    HW_OK=1
fi

echo ""
echo "Status:"
if [ $PYTHON_OK -eq 1 ]; then
    echo "  ✓ Python 3.12+"
else
    echo "  ✗ Python 3.12+ (need upgrade)"
fi

if [ $TTMETAL_OK -eq 1 ]; then
    echo "  ✓ tt-metal"
else
    echo "  ✗ tt-metal (not installed)"
fi

if [ $FORGE_OK -eq 1 ]; then
    echo "  ✓ tt-forge-fe"
else
    echo "  ✗ tt-forge-fe (not installed)"
fi

if [ $HW_OK -eq 1 ]; then
    echo "  ✓ Hardware detected"
else
    echo "  ⚠  No hardware detected"
fi

# Next steps
echo ""
echo "Next Steps:"
echo ""

if [ $FORGE_OK -eq 0 ]; then
    echo "1. Install tt-forge-fe:"
    echo "   python3 $SCRIPT_DIR/compiletron.py setup install-forge"
    echo ""
fi

if [ $FORGE_OK -eq 1 ]; then
    echo "1. Activate Forge environment:"
    echo "   source $FORGE_HOME/env/activate"
    echo ""
fi

echo "2. Explore the CLI:"
echo "   python3 $SCRIPT_DIR/compiletron.py detect"
echo "   python3 $SCRIPT_DIR/compiletron.py models stats"
echo "   python3 $SCRIPT_DIR/compiletron.py models quick"
echo ""

echo "3. Check full status:"
echo "   python3 $SCRIPT_DIR/compiletron.py setup check"
echo ""

if [ $FORGE_OK -eq 1 ] && [ $HW_OK -eq 1 ]; then
    echo "✅ Ready to compile models!"
    echo "   python3 $SCRIPT_DIR/compiletron.py run --quick"
else
    echo "⚠️  Complete setup steps above before running models"
fi

echo ""
