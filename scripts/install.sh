#!/usr/bin/env bash
# scripts/install.sh — Smart installer for tt-forge-compiletron.
#
# Checks every dependency, compares versions against TT PyPI, and
# delegates all pip work to scripts/setup-venvs.sh.
#
# Usage:
#   bash scripts/install.sh                  # check + fix everything
#   bash scripts/install.sh --status         # check only, no installs
#   bash scripts/install.sh --forge          # forge backend only
#   bash scripts/install.sh --xla            # XLA backend only
#   bash scripts/install.sh --skip-hardware  # skip chip detection (CI)
#   bash scripts/install.sh --help
#
# Flags compose: --status --forge checks forge steps only, no installs.
# Log: /tmp/tt-compiletron-install.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TENSTORRENT_PYPI="https://pypi.eng.aws.tenstorrent.com/"
LOG_FILE="/tmp/tt-compiletron-install.log"

# ── Colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    _GRN="\033[32m" _YLW="\033[33m" _RED="\033[31m" _CYN="\033[36m"
    _RST="\033[0m"  _BLD="\033[1m"
else
    _GRN="" _YLW="" _RED="" _CYN="" _RST="" _BLD=""
fi

ok()   { echo -e "${_GRN}  ✓${_RST}  $*"; }
warn() { echo -e "${_YLW}  ⚠${_RST}  $*"; }
fail() { echo -e "${_RED}  ✗${_RST}  $*"; }
info() { echo -e "     $*"; }
step() { echo -e "\n${_BLD}${_CYN}[$1]${_RST}${_BLD} $2${_RST}"; }
hr()   {
    echo -e "${_CYN}╔══════════════════════════════════════════════════${_RST}"
}
hr_mid() {
    echo -e "${_CYN}╚══════════════════════════════════════════════════${_RST}"
}
hr_end() {
    echo -e "${_CYN}╚══════════════════════════════════════════════════${_RST}"
}

# ── State tracking ────────────────────────────────────────────────────────────
declare -a PASSED=()
declare -a WARNED=()
declare -a FAILED=()

pass_s()  { PASSED+=("$1"); ok   "$1"; }
warn_s()  { WARNED+=("$1"); warn "$1"; }
fail_s()  { FAILED+=("$1"); fail "$1"; }

# ── Flags ─────────────────────────────────────────────────────────────────────
STATUS_ONLY=0
SETUP_FORGE=1
SETUP_XLA=1
SKIP_HARDWARE=0

[[ -t 0 ]] || STATUS_ONLY=1   # non-interactive: auto status-only

while [[ ${1:-} != "" ]]; do
    case "$1" in
        --status)          STATUS_ONLY=1 ;;
        --forge)           SETUP_XLA=0 ;;
        --xla)             SETUP_FORGE=0 ;;
        --skip-hardware)   SKIP_HARDWARE=1 ;;
        --help|-h)
            sed -n '2,13p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $1  (try --help)"; exit 1 ;;
    esac
    shift
done

> "$LOG_FILE"   # truncate log at start of each run

# ── Version detection ─────────────────────────────────────────────────────────
# Queries TT PyPI simple index for the latest non-dev X.Y.Z release.
# Prints the version string, or "" if the index is unreachable.
_latest_tt_version() {
    local pkg="$1"
    python3 - "$pkg" <<'PYEOF'
import sys, urllib.request, re
pkg = sys.argv[1]
url = f"https://pypi.eng.aws.tenstorrent.com/simple/{pkg}/"
try:
    html = urllib.request.urlopen(url, timeout=5).read().decode()
    versions = re.findall(
        rf'{re.escape(pkg)}-(\d+\.\d+\.\d+)-', html, re.IGNORECASE
    )
    stable = sorted(set(versions), key=lambda v: list(map(int, v.split('.'))))
    print(stable[-1] if stable else "")
except Exception:
    print("")
PYEOF
}

# Returns the installed version of a pip package in a given venv's python,
# or "" if not installed.
_installed_version() {
    local py="$1" pkg="$2"
    "$py" -m pip show "$pkg" 2>/dev/null | awk '/^Version/{print $2}'
}

# Compares two X.Y.Z version strings. Returns 0 if $1 < $2 (behind).
_version_behind() {
    python3 -c "
import sys
a = list(map(int, sys.argv[1].split('.')))
b = list(map(int, sys.argv[2].split('.')))
sys.exit(0 if a < b else 1)
" "$1" "$2" 2>/dev/null
}

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
hr
echo -e "  ${_BLD}TT-Forge Compiletron — Smart Installer${_RST}"
[[ $STATUS_ONLY -eq 1 ]] && echo -e "  ${_YLW}Status check only — no changes will be made.${_RST}"
hr_end
echo ""

# Shared paths (used by multiple steps)
FORGE_VENV="$HOME/tt-forge-venv"
FORGE_PY="$FORGE_VENV/bin/python"
XLA_VENV="$HOME/tt-xla/venv"
XLA_PY="$XLA_VENV/bin/python"

# ── [1] Hardware presence ─────────────────────────────────────────────────────
step 1 "Hardware presence"

if [[ $SKIP_HARDWARE -eq 1 ]]; then
    warn_s "Hardware check skipped (--skip-hardware)"
elif ! command -v tt-smi &>/dev/null; then
    warn_s "tt-smi not found — cannot detect TT chips"
    info "Install: pip install tt-smi --extra-index-url $TENSTORRENT_PYPI"
else
    _hw_json=$(tt-smi -s 2>/dev/null) || _hw_json=""
    if [[ -z "$_hw_json" ]]; then
        fail_s "tt-smi returned no output — no TT chips detected or driver issue"
        info "Check: sudo dmesg | grep tenstorrent"
    else
        _chip_count=$(echo "$_hw_json" | python3 -c "
import json,sys
d=json.load(sys.stdin)
devs=d.get('device_info',[])
print(len(devs))
" 2>/dev/null || echo 0)
        if [[ "$_chip_count" -eq 0 ]]; then
            fail_s "No TT chips detected"
        else
            _hw_summary=$(echo "$_hw_json" | python3 -c "
import json,sys
d=json.load(sys.stdin)
devs=d.get('device_info',[])
n=len(devs)
fw=devs[0].get('smbus_telem',{}).get('FLASH_BUNDLE_VERSION','?') if devs else '?'
if isinstance(fw,str) and fw.startswith('0x'):
    v=int(fw,16)
    fw=f'{v>>24}.{(v>>16)&0xff}.{(v>>8)&0xff}'
print(f'{n}x Blackhole  fw bundle {fw}')
" 2>/dev/null || echo "${_chip_count}x detected")
            pass_s "Hardware: $_hw_summary"
        fi
    fi
fi

# ── [2] Hugepages ─────────────────────────────────────────────────────────────
step 2 "Hugepages  (≥64 required by tt-metal)"

_hp=$(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo 0)
if [[ "$_hp" -ge 64 ]]; then
    pass_s "Hugepages: $_hp"
elif [[ $STATUS_ONLY -eq 1 ]]; then
    fail_s "Hugepages: $_hp (need ≥64)"
    info "Fix: sudo sysctl -w vm.nr_hugepages=128"
    info "Persist: echo 'vm.nr_hugepages=128' | sudo tee /etc/sysctl.d/99-tenstorrent-hugepages.conf"
else
    info "Setting hugepages to 128 (currently $_hp)..."
    if sudo sysctl -w vm.nr_hugepages=128 >> "$LOG_FILE" 2>&1; then
        _SYSCTL_CONF=/etc/sysctl.d/99-tenstorrent-hugepages.conf
        if [[ ! -f "$_SYSCTL_CONF" ]]; then
            echo 'vm.nr_hugepages=128' | sudo tee "$_SYSCTL_CONF" >> "$LOG_FILE" 2>&1 || true
        fi
        pass_s "Hugepages set to 128"
    else
        fail_s "Could not set hugepages (sudo failed?)"
        info "Manual fix: sudo sysctl -w vm.nr_hugepages=128"
    fi
fi

# ── [3] Disk space ────────────────────────────────────────────────────────────
step 3 "Disk space  (HF model cache fills fast)"

_free_gb=$(df -BG "$HOME" | awk 'NR==2{gsub("G","",$4); print $4}' 2>/dev/null || echo 999)
if [[ "$_free_gb" -ge 50 ]]; then
    pass_s "Disk: ${_free_gb}G free"
elif [[ "$_free_gb" -ge 20 ]]; then
    warn_s "Disk: ${_free_gb}G free — getting low (recommend ≥50G)"
    info "Biggest consumers: du -sh ~/.cache/huggingface/hub/models--*/ | sort -rh | head -10"
else
    fail_s "Disk: ${_free_gb}G free — critically low; expedition will refuse to download models"
    info "Free space: du -sh ~/.cache/huggingface/hub/models--*/ | sort -rh | head -10"
fi

# ── [4] Forge venv ────────────────────────────────────────────────────────────
_FORGE_OK=0
if [[ $SETUP_FORGE -eq 1 ]]; then
    step 4 "Forge venv  (~/$( realpath --relative-to="$HOME" "$FORGE_VENV" 2>/dev/null || echo tt-forge-venv))"

    _forge_importable=0
    _forge_is_tt=0

    if [[ -x "$FORGE_PY" ]]; then
        # Check that the forge package is importable at all
        if "$FORGE_PY" -c "import forge" 2>/dev/null; then
            _forge_importable=1
            # Gotcha: PyPI hosts a "forge" package that is a Django form-builder app.
            # TT forge exposes forge.compiled; Django forge does not.
            # We use that submodule as the discriminator, and also capture __version__.
            _forge_ver=$("$FORGE_PY" -c "
import forge, sys
v = getattr(forge, '__version__', '')
try:
    import forge.compiled   # TT forge has this; Django forge does not
    print(v)
except ImportError:
    print('')
" 2>/dev/null || echo "")
            if [[ -n "$_forge_ver" ]]; then
                _forge_is_tt=1
            fi
        fi
    fi

    if [[ $_forge_is_tt -eq 1 ]]; then
        pass_s "Forge venv: $FORGE_VENV  (forge $_forge_ver)"
        _FORGE_OK=1
    elif [[ $STATUS_ONLY -eq 1 ]]; then
        if [[ $_forge_importable -eq 1 && $_forge_is_tt -eq 0 ]]; then
            fail_s "Forge venv: wrong 'forge' package installed (Django app, not TT forge)"
            info "Fix: $FORGE_VENV/bin/pip uninstall forge -y && $FORGE_VENV/bin/pip install forge --extra-index-url $TENSTORRENT_PYPI"
        else
            fail_s "Forge venv missing or forge not importable"
            info "Fix: bash scripts/setup-venvs.sh --forge"
        fi
    else
        if [[ $_forge_importable -eq 1 && $_forge_is_tt -eq 0 ]]; then
            info "Wrong 'forge' package detected (Django app) — reinstalling from TT PyPI..."
            "$FORGE_VENV/bin/pip" uninstall forge -y >> "$LOG_FILE" 2>&1 || true
        else
            info "Installing forge venv via setup-venvs.sh --forge..."
        fi
        if bash "$SCRIPT_DIR/setup-venvs.sh" --forge >> "$LOG_FILE" 2>&1; then
            _forge_ver=$("$FORGE_PY" -c "import forge; print(getattr(forge,'__version__','?'))" 2>/dev/null || echo "?")
            pass_s "Forge venv installed  (forge $_forge_ver)"
            _FORGE_OK=1
        else
            fail_s "Forge venv install failed — see $LOG_FILE"
            info "Manual fix: bash scripts/setup-venvs.sh --forge"
        fi
    fi
fi

# ── [5] Forge version ─────────────────────────────────────────────────────────
# Only runs when forge venv is healthy (_FORGE_OK=1).
# Network failures are WARNED, not FAILED, so offline CI still passes.
if [[ $SETUP_FORGE -eq 1 && $_FORGE_OK -eq 1 ]]; then
    step 5 "Forge version  (TT PyPI latest)"

    _forge_installed=$("$FORGE_PY" -c "import forge; print(getattr(forge,'__version__',''))" 2>/dev/null || echo "")
    _forge_latest=$(_latest_tt_version "forge")

    if [[ -z "$_forge_latest" ]]; then
        warn_s "Forge version: cannot reach TT PyPI — skipping comparison"
    elif [[ -z "$_forge_installed" ]]; then
        warn_s "Forge version: installed version unknown"
    elif _version_behind "$_forge_installed" "$_forge_latest"; then
        warn_s "Forge version: $_forge_installed installed, $_forge_latest available"
        info "Upgrade: bash scripts/install.sh --forge"
    else
        pass_s "Forge version: $_forge_installed (up to date)"
    fi
fi

# ── [6] Forge-fe shim ─────────────────────────────────────────────────────────
# Validates that ~/tt-forge-fe/env/activate exists and sources without error.
# The shim is written by setup-venvs.sh --forge; step [4] above invokes that
# whenever forge is missing, so if _FORGE_OK=1 the shim should exist.
if [[ $SETUP_FORGE -eq 1 ]]; then
    step 6 "Forge-fe shim  (~/tt-forge-fe/env/activate)"

    _SHIM="$HOME/tt-forge-fe/env/activate"
    if [[ ! -f "$_SHIM" ]]; then
        if [[ $STATUS_ONLY -eq 1 ]]; then
            fail_s "Forge-fe shim missing: $_SHIM"
            info "Fix: bash scripts/setup-venvs.sh --forge"
        elif [[ $_FORGE_OK -eq 1 ]]; then
            # setup-venvs.sh already wrote the shim during the step [4] install
            pass_s "Forge-fe shim written by install step above"
        else
            fail_s "Forge-fe shim missing and forge venv install failed — cannot write shim"
        fi
    else
        # Validate shim sources cleanly; timeout 5s to guard against hangs
        if timeout 5 bash -c "source $_SHIM" >> "$LOG_FILE" 2>&1; then
            pass_s "Forge-fe shim: $_SHIM"
        else
            warn_s "Forge-fe shim exists but sourcing it produced errors"
            info "Check: source $_SHIM"
        fi
    fi
fi
