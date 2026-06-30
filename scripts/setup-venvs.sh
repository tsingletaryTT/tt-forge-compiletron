#!/usr/bin/env bash
# scripts/setup-venvs.sh
# Turn-key environment setup for tt-forge-compiletron on Ubuntu 24.04.
#
# Creates two Python 3.12 venvs from Tenstorrent pip wheels:
#
#   ~/tt-forge-venv   -- forge backend  (tt-forge PyTorch compiler)
#   ~/tt-xla/venv     -- XLA backend    (pjrt-plugin-tt JAX compiler)
#
# Also writes a ~/tt-forge-fe/env/activate shim so the expedition harness
# finds the forge venv without requiring a source build of tt-forge-fe.
#
# Usage:
#   bash scripts/setup-venvs.sh             # both backends
#   bash scripts/setup-venvs.sh --forge     # forge only
#   bash scripts/setup-venvs.sh --xla       # XLA only
#   bash scripts/setup-venvs.sh --no-smi    # skip tt-smi install
#
# Re-running is safe: existing venvs are upgraded in place.

set -euo pipefail

TENSTORRENT_PYPI="https://pypi.eng.aws.tenstorrent.com/"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────

SETUP_FORGE=1
SETUP_XLA=1
SETUP_SMI=1

for arg in "$@"; do
    case "$arg" in
        --forge)   SETUP_XLA=0   ;;
        --xla)     SETUP_FORGE=0 ;;
        --no-smi)  SETUP_SMI=0   ;;
        --help|-h)
            sed -n '2,/^set /p' "$0" | grep '^#' | sed 's/^# *//'
            exit 0
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

ok()   { echo "[ ok ] $*"; }
info() { echo "       $*"; }
warn() { echo "[warn] $*"; }
fail() { echo "[fail] $*"; exit 1; }

header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════"
    echo "║  $*"
    echo "╚══════════════════════════════════════════════════════"
}

# ── 1. Prerequisites ──────────────────────────────────────────────────────────

header "System prerequisites"

if ! python3.12 --version &>/dev/null; then
    info "Python 3.12 not found — installing..."
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
fi
PY="python3.12"
ok "Python: $($PY --version)"

for pkg in build-essential git; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        info "Installing $pkg..."
        sudo apt-get install -y "$pkg"
    fi
done
ok "Build tools + git present"

# tmux is a convenience for running expeditions, not a hard dependency
if ! command -v tmux &>/dev/null; then
    if sudo apt-get install -y tmux &>/dev/null 2>&1; then
        ok "tmux installed"
    else
        warn "tmux not available in apt — install manually if needed for expedition sessions"
    fi
fi

# ── 2. Hugepages ──────────────────────────────────────────────────────────────

header "Hugepages (required by tt-metal)"

CURRENT_HP=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo 0)
if [[ "$CURRENT_HP" -lt 64 ]]; then
    info "Setting hugepages to 128 (currently $CURRENT_HP)..."
    sudo sysctl -w vm.nr_hugepages=128
    SYSCTL_CONF=/etc/sysctl.d/99-tenstorrent-hugepages.conf
    if [[ ! -f "$SYSCTL_CONF" ]]; then
        echo 'vm.nr_hugepages=128' | sudo tee "$SYSCTL_CONF" > /dev/null
        ok "Persisted to $SYSCTL_CONF"
    fi
else
    ok "Hugepages already set ($CURRENT_HP)"
fi

# ── 3. tt-smi ─────────────────────────────────────────────────────────────────

if [[ "$SETUP_SMI" -eq 1 ]]; then
    header "tt-smi (hardware monitor)"

    if command -v tt-smi &>/dev/null; then
        ok "tt-smi already installed: $(tt-smi --version 2>/dev/null || echo 'present')"
    else
        info "Installing tt-smi..."
        pip install tt-smi --extra-index-url "$TENSTORRENT_PYPI" -q
        ok "tt-smi installed"
    fi
fi

# ── 4. Forge venv (~/tt-forge-venv) ──────────────────────────────────────────

if [[ "$SETUP_FORGE" -eq 1 ]]; then
    header "Forge backend venv  (~/.tt-forge-venv)"

    FORGE_VENV="$HOME/tt-forge-venv"

    if [[ ! -d "$FORGE_VENV" ]]; then
        info "Creating $FORGE_VENV ..."
        "$PY" -m venv "$FORGE_VENV"
    fi

    info "Installing forge wheel..."
    "$FORGE_VENV/bin/pip" install -q --upgrade pip
    # Install forge with TT PyPI as the primary index so the Tenstorrent forge
    # wheel is resolved instead of the unrelated Django "forge" package on public PyPI.
    # --index-url makes TT PyPI primary; --extra-index-url adds public PyPI as fallback.
    "$FORGE_VENV/bin/pip" install -q \
        forge \
        --index-url "$TENSTORRENT_PYPI" \
        --extra-index-url "https://pypi.org/simple/"

    info "Installing compiletron application deps..."
    "$FORGE_VENV/bin/pip" install -q -r "$PROJECT_DIR/requirements.txt"

    info "Installing optional embedding-model deps..."
    "$FORGE_VENV/bin/pip" install -q FlagEmbedding tf-keras

    ok "Forge venv ready: $FORGE_VENV"
    ok "forge version: $("$FORGE_VENV/bin/python" -c 'import forge; print(forge.__version__)' 2>/dev/null || echo 'import ok')"

    # ── 4a. ~/tt-forge-fe/env/activate shim ─────────────────────────────────

    SHIM_DIR="$HOME/tt-forge-fe/env"
    SHIM="$SHIM_DIR/activate"

    # Only write the shim if the real tt-forge-fe source tree is absent — if it
    # exists the genuine activate script is already there and should be used.
    if [[ -f "$HOME/tt-forge-fe/env/activate" ]]; then
        ok "Existing ~/tt-forge-fe/env/activate found — leaving it unchanged"
    else
        info "Writing ~/tt-forge-fe/env/activate shim (pip-wheel path)..."
        mkdir -p "$SHIM_DIR"
        cat > "$SHIM" << 'SHIM_SCRIPT'
#!/usr/bin/env bash
# ~/tt-forge-fe/env/activate — pip-wheel shim written by setup-venvs.sh.
# Points at ~/tt-forge-venv instead of a source-built toolchain.
# The forge wheel is self-contained; only ARCH_NAME and the guard vars need to
# be set so the harness (run_expedition.sh) recognises the env as activated.

export TTFORGE_TOOLCHAIN_DIR="${TTFORGE_TOOLCHAIN_DIR:-${HOME}/tt-forge-venv}"
export TTFORGE_PYTHON_VERSION="${TTFORGE_PYTHON_VERSION:-python3.12}"
export TTFORGE_VENV_DIR="${TTFORGE_VENV_DIR:-${HOME}/tt-forge-venv}"
export TTMLIR_TOOLCHAIN_DIR="${TTMLIR_TOOLCHAIN_DIR:-${HOME}/tt-forge-venv}"
export TTMLIR_VENV_DIR="${TTMLIR_VENV_DIR:-${HOME}/tt-forge-venv}"
export TTMLIR_ENV_ACTIVATED=1

# Blackhole is the default; override if running on Wormhole hardware.
#   export ARCH_NAME=wormhole_b0   (Wormhole boards: n150, n300)
#   export ARCH_NAME=grayskull     (Grayskull boards: e75, e150)
export ARCH_NAME="${ARCH_NAME:-blackhole}"

# Allow uv to resolve across both PyPI and Tenstorrent indexes.
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"

# Activate the forge venv.
# shellcheck source=/dev/null
source "${HOME}/tt-forge-venv/bin/activate"
SHIM_SCRIPT
        chmod +x "$SHIM"
        ok "Shim written to $SHIM"
    fi
fi

# ── 5. XLA venv (~/tt-xla/venv) ──────────────────────────────────────────────

if [[ "$SETUP_XLA" -eq 1 ]]; then
    header "XLA backend venv  (~/tt-xla/venv)"

    XLA_VENV="$HOME/tt-xla/venv"

    if [[ ! -d "$XLA_VENV" ]]; then
        info "Creating $XLA_VENV ..."
        mkdir -p "$HOME/tt-xla"
        "$PY" -m venv "$XLA_VENV"
    fi

    info "Installing pjrt-plugin-tt and JAX stack..."
    "$XLA_VENV/bin/pip" install -q --upgrade pip

    # Install torch + torchvision from the PyTorch CPU wheel index so both get
    # ABI-matched +cpu builds.  Resolving torchvision via public PyPI produces a
    # CUDA-flavoured build that fails to load against the CPU torch
    # (RuntimeError: operator torchvision::nms does not exist).
    # timm is pure-Python and only exists on PyPI — install it separately.
    "$XLA_VENV/bin/pip" install -q \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

    "$XLA_VENV/bin/pip" install -q \
        pjrt-plugin-tt \
        jax jaxlib \
        flax \
        "transformers<5.0" \
        timm \
        pyfiglet \
        --extra-index-url "$TENSTORRENT_PYPI"

    ok "XLA venv ready: $XLA_VENV"
    ok "pjrt-plugin-tt: $("$XLA_VENV/bin/pip" show pjrt-plugin-tt 2>/dev/null | awk '/^Version/{print $2}')"
    ok "jax: $("$XLA_VENV/bin/python" -c 'import jax; print(jax.__version__)' 2>/dev/null)"

    # ── 5a. Mesh graph descriptor ─────────────────────────────────────────────
    #
    # The XLA worker auto-detects the descriptor from ~/tt-xla at startup.
    # tt-xla source builds place it at:
    #   ~/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/
    #       mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
    #
    # If ~/tt-xla is a plain venv directory (pip-wheel path), the auto-detect
    # fails.  In that case we symlink the bundled descriptor from this repo:

    TT_XLA_DESCRIPTOR_PATH="$HOME/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors"
    BUNDLED_DESCRIPTOR="$PROJECT_DIR/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"

    if [[ ! -f "$TT_XLA_DESCRIPTOR_PATH/p100_mesh_graph_descriptor.textproto" ]]; then
        info "XLA mesh descriptor not found in ~/tt-xla tree — linking bundled descriptor..."
        mkdir -p "$TT_XLA_DESCRIPTOR_PATH"
        ln -sf "$BUNDLED_DESCRIPTOR" "$TT_XLA_DESCRIPTOR_PATH/p100_mesh_graph_descriptor.textproto"
        ok "Linked $BUNDLED_DESCRIPTOR"
    else
        ok "XLA mesh descriptor found in ~/tt-xla tree"
    fi
fi

# ── 6. tt-forge-models (seed model zoo) ──────────────────────────────────────

header "tt-forge-models seed zoo"

MODELS_DIR="$HOME/code/tt-forge-models"
if [[ -d "$MODELS_DIR" ]]; then
    ok "tt-forge-models found at $MODELS_DIR"
else
    info "Cloning tt-forge-models into $MODELS_DIR ..."
    mkdir -p "$(dirname "$MODELS_DIR")"
    if git clone --depth=1 https://github.com/tenstorrent/tt-forge-models.git "$MODELS_DIR"; then
        ok "tt-forge-models cloned at $MODELS_DIR"
    else
        warn "tt-forge-models clone failed — frontier-only mode still works"
        info "Retry manually: git clone https://github.com/tenstorrent/tt-forge-models.git $MODELS_DIR"
    fi
fi

# ── 7. Summary ────────────────────────────────────────────────────────────────

header "Setup complete"

echo ""
echo "╔══════════════════════════════════════════════════════"
echo "║  Quick start"
echo "╠══════════════════════════════════════════════════════"

if [[ "$SETUP_FORGE" -eq 1 ]]; then
echo "║"
echo "║  Forge backend:"
echo "║    source ~/tt-forge-fe/env/activate"
echo "║    python3 expedition.py run --tui"
echo "║"
echo "║  Or let expedition.py activate the env automatically:"
echo "║    python3 expedition.py run --tui"
fi

if [[ "$SETUP_XLA" -eq 1 ]]; then
echo "║"
echo "║  XLA backend (uses ~/tt-xla/venv automatically):"
echo "║    python3 expedition.py run --tui --backend xla"
echo "║"
echo "║  Mixed (even chips=forge, odd=xla):"
echo "║    python3 expedition.py run --tui --backend mixed"
fi

echo "║"
echo "╚══════════════════════════════════════════════════════"
echo ""
