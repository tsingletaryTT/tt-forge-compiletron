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
_hw_json=""   # populated by step 1; read by step 8 firmware probe

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

# ── [7] XLA venv ──────────────────────────────────────────────────────────────
_XLA_OK=0
if [[ $SETUP_XLA -eq 1 ]]; then
    step 7 "XLA venv  (~/tt-xla/venv)"

    _xla_importable=0
    if [[ -x "$XLA_PY" ]]; then
        if "$XLA_PY" -c "import pjrt_plugin_tt" 2>/dev/null; then
            _xla_importable=1
        fi
    fi

    if [[ $_xla_importable -eq 1 ]]; then
        _pjrt_ver=$(_installed_version "$XLA_PY" "pjrt-plugin-tt")
        # Warn if it's a dev build (contains '+dev') — explicit pinning was needed
        if [[ "$_pjrt_ver" == *"+dev"* ]]; then
            warn_s "XLA venv: pjrt-plugin-tt $_pjrt_ver (dev build — explicit pin recommended)"
            info "Upgrade: $XLA_VENV/bin/pip install pjrt-plugin-tt==<latest> --extra-index-url $TENSTORRENT_PYPI"
            _XLA_OK=1   # functional, just not ideal
        else
            pass_s "XLA venv: $XLA_VENV  (pjrt-plugin-tt $_pjrt_ver)"
            _XLA_OK=1
        fi

        # Note harmless triton/easydel conflict — reassure the user
        if "$XLA_PY" -m pip check 2>/dev/null | grep -q "easydel\|triton" 2>/dev/null; then
            info "Note: triton/easydel dep conflict in XLA venv — harmless, XLA compile still works"
        fi
    elif [[ $STATUS_ONLY -eq 1 ]]; then
        fail_s "XLA venv missing or pjrt-plugin-tt not importable"
        info "Fix: bash scripts/setup-venvs.sh --xla"
    else
        info "Installing XLA venv via setup-venvs.sh --xla..."
        if bash "$SCRIPT_DIR/setup-venvs.sh" --xla >> "$LOG_FILE" 2>&1; then
            _pjrt_ver=$(_installed_version "$XLA_PY" "pjrt-plugin-tt")
            pass_s "XLA venv installed  (pjrt-plugin-tt $_pjrt_ver)"
            _XLA_OK=1
        else
            fail_s "XLA venv install failed — see $LOG_FILE"
            info "Manual fix: bash scripts/setup-venvs.sh --xla"
        fi
    fi
fi

# ── [8] XLA version + device init probe ──────────────────────────────────────
if [[ $SETUP_XLA -eq 1 && $_XLA_OK -eq 1 ]]; then
    step 8 "XLA version  (pjrt-plugin-tt) + device init probe"

    _pjrt_installed=$(_installed_version "$XLA_PY" "pjrt-plugin-tt")
    _pjrt_installed_clean="${_pjrt_installed%%+*}"
    _pjrt_latest=$(_latest_tt_version "pjrt-plugin-tt")

    # Version comparison (advisory)
    if [[ -z "$_pjrt_latest" ]]; then
        warn_s "XLA version: cannot reach TT PyPI — skipping comparison"
    elif [[ -z "$_pjrt_installed_clean" ]]; then
        warn_s "XLA version: installed version unknown"
    elif _version_behind "$_pjrt_installed_clean" "$_pjrt_latest"; then
        info "XLA version: $_pjrt_installed installed, $_pjrt_latest stable available"
    else
        pass_s "XLA version: $_pjrt_installed (up to date)"
    fi

    # Device init probe — catches firmware/pjrt-plugin-tt mismatches before
    # expedition time.  pjrt-plugin-tt stable releases can require newer
    # tt-metal firmware than the board bundle carries; a dev build from the
    # matching date avoids this.
    info "Probing device init (JAX_PLATFORMS=tt jax.devices())..."
    _probe_out=$(JAX_PLATFORMS=tt TT_METAL_LOGGER_LEVEL=FATAL "$XLA_PY" -c "
import os, sys
os.environ.setdefault('TT_METAL_LOGGER_LEVEL', 'FATAL')
import jax
try:
    devs = jax.devices()
    print(f'ok:{len(devs)}')
except Exception as e:
    sys.stderr.write(f'fail:{e}\n')
    sys.exit(1)
" 2>/tmp/tt-xla-probe.err)
    _probe_rc=$?

    if [[ $_probe_rc -eq 0 ]]; then
        _probe_count="${_probe_out#ok:}"
        pass_s "XLA device init: ${_probe_count} device(s) detected"
    else
        _probe_err=$(cat /tmp/tt-xla-probe.err 2>/dev/null | head -2)
        warn_s "XLA device init failed: $_probe_err"
        info "This usually means pjrt-plugin-tt was built against newer firmware than your board carries."

        # Try to find a dev build whose build date predates the current firmware
        # bundle date.  TT dev build names follow: 1.2.0.devYYYYMMDD######
        # Firmware bundle FLASH_BUNDLE_VERSION 19.11.0 was current around 2026-05-28.
        _fw_ver=$(echo "$_hw_json" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
devs=d.get('device_info',[])
fw=devs[0].get('smbus_telem',{}).get('FLASH_BUNDLE_VERSION','') if devs else ''
if isinstance(fw,str) and fw.startswith('0x'):
    v=int(fw,16)
    fw=f'{v>>24}.{(v>>16)&0xff}.{(v>>8)&0xff}'
print(fw)
" 2>/dev/null || echo "")

        _compatible_dev=$(python3 - "$_fw_ver" <<'PYEOF'
import sys, urllib.request, re
fw_ver = sys.argv[1] if len(sys.argv) > 1 else ""

# Map known fw bundles to the latest pjrt-plugin-tt dev build that works with them.
# Key is the fw bundle version string; value is the dev build version to pin.
FW_COMPAT = {
    "19.11.0": "1.2.0.dev20260528002737",
}

if fw_ver in FW_COMPAT:
    print(FW_COMPAT[fw_ver])
    sys.exit(0)

# Unknown firmware — try to find the newest dev build older than current stable
url = "https://pypi.eng.aws.tenstorrent.com/simple/pjrt-plugin-tt/"
try:
    html = urllib.request.urlopen(url, timeout=5).read().decode()
    devs = re.findall(r'pjrt.plugin.tt-(1\.2\.0\.dev\d{14})-', html)
    if devs:
        devs.sort()
        print(devs[-2] if len(devs) > 1 else devs[0])
except Exception:
    pass
PYEOF
        )

        if [[ -n "$_compatible_dev" ]]; then
            info "Known-compatible dev build for fw ${_fw_ver:-unknown}: $_compatible_dev"
            if [[ $STATUS_ONLY -eq 0 ]]; then
                info "Installing pjrt-plugin-tt==$_compatible_dev ..."
                if "$XLA_VENV/bin/pip" install "pjrt-plugin-tt==$_compatible_dev" \
                        --extra-index-url "$TENSTORRENT_PYPI" -q >> "$LOG_FILE" 2>&1; then
                    # Re-probe
                    _reprobe=$(JAX_PLATFORMS=tt TT_METAL_LOGGER_LEVEL=FATAL "$XLA_PY" -c "
import os, sys
os.environ.setdefault('TT_METAL_LOGGER_LEVEL', 'FATAL')
import jax
try:
    devs = jax.devices()
    print(f'ok:{len(devs)}')
except Exception as e:
    sys.exit(1)
" 2>/dev/null)
                    if [[ "$_reprobe" == ok:* ]]; then
                        pass_s "XLA device init: ${_reprobe#ok:} device(s) after downgrade to $_compatible_dev"
                    else
                        warn_s "XLA device init still failing after downgrade — check firmware bundle"
                    fi
                else
                    warn_s "pjrt-plugin-tt downgrade failed — see $LOG_FILE"
                fi
            else
                info "Fix (--status mode): $XLA_VENV/bin/pip install pjrt-plugin-tt==$_compatible_dev --extra-index-url $TENSTORRENT_PYPI"
            fi
        else
            info "Fix: pin pjrt-plugin-tt to a dev build matching your firmware bundle."
            info "Check: $XLA_VENV/bin/pip install pjrt-plugin-tt==<dev-build> --extra-index-url $TENSTORRENT_PYPI"
        fi
    fi
fi

# ── [9] Mesh descriptor ───────────────────────────────────────────────────────
if [[ $SETUP_XLA -eq 1 ]]; then
    step 9 "XLA mesh descriptor  (p100_mesh_graph_descriptor.textproto)"

    _MESH_DIR="$HOME/tt-xla/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors"
    _MESH_FILE="$_MESH_DIR/p100_mesh_graph_descriptor.textproto"
    _BUNDLED="$PROJECT_DIR/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"

    if [[ -f "$_MESH_FILE" ]]; then
        pass_s "Mesh descriptor: present"
    elif [[ $STATUS_ONLY -eq 1 ]]; then
        fail_s "Mesh descriptor missing: $_MESH_FILE"
        info "Fix: bash scripts/setup-venvs.sh --xla  (links the bundled descriptor)"
        info "Or:  mkdir -p $_MESH_DIR && ln -sf $_BUNDLED $_MESH_FILE"
    else
        if [[ ! -f "$_BUNDLED" ]]; then
            fail_s "Mesh descriptor: bundled file not found at $_BUNDLED"
            info "Is the project directory intact? Expected: $PROJECT_DIR/mesh_graph_descriptors/"
        else
            info "Linking bundled mesh descriptor..."
            mkdir -p "$_MESH_DIR"
            ln -sf "$_BUNDLED" "$_MESH_FILE"
            pass_s "Mesh descriptor linked"
        fi
    fi
fi

# ── [10] tt-forge-models ──────────────────────────────────────────────────────
step 10 "tt-forge-models  (seed model zoo)"

_MODELS_DIR="$HOME/code/tt-forge-models"
if [[ -d "$_MODELS_DIR" ]]; then
    _model_count=$(find "$_MODELS_DIR" -maxdepth 2 -name "*.py" 2>/dev/null | wc -l)
    pass_s "tt-forge-models: $_MODELS_DIR  (~$_model_count py files)"
else
    warn_s "tt-forge-models not found at $_MODELS_DIR"
    info "Seed model runs need it. Clone:"
    info "  git clone https://github.com/tenstorrent/tt-forge-models.git $_MODELS_DIR"
    info "Frontier-only mode works without it: python3 expedition.py run --frontier-only"
fi

# ── [11] Stale /dev/shm segments ─────────────────────────────────────────────
step 11 "Stale /dev/shm segments  (leftover from crashed runs)"

_stale=$(find /dev/shm -maxdepth 1 -name "sm_segment.tt-quietbox.*.0" 2>/dev/null)
if [[ -z "$_stale" ]]; then
    pass_s "No stale /dev/shm segments"
else
    _count=$(echo "$_stale" | wc -l)
    warn_s "$_count stale segment(s) in /dev/shm — leftover from a crashed expedition"
    info "These can cause 'address already in use' errors on next run."
    info "Clean up: find /dev/shm -name 'sm_segment.tt-quietbox.*.0' -delete"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
hr
echo -e "  ${_BLD}Summary${_RST}"
hr_mid

for s in "${PASSED[@]:-}";  do [[ -n "$s" ]] && echo -e "${_GRN}  ✓${_RST}  $s"; done
for s in "${WARNED[@]:-}";  do [[ -n "$s" ]] && echo -e "${_YLW}  ⚠${_RST}  $s"; done
for s in "${FAILED[@]:-}";  do [[ -n "$s" ]] && echo -e "${_RED}  ✗${_RST}  $s"; done

hr_end
echo ""

if [[ ${#FAILED[@]} -eq 0 ]]; then
    if [[ $STATUS_ONLY -eq 1 ]]; then
        echo -e "${_GRN}  All checks passed.${_RST}"
    else
        echo -e "${_GRN}  ${_BLD}Ready!${_RST}  Run an expedition:"
        echo -e "  Forge:  python3 expedition.py run --tui"
        echo -e "  XLA:    python3 expedition.py run --tui --backend xla"
        echo -e "  Mixed:  python3 expedition.py run --tui --backend mixed"
    fi
    echo ""
    exit 0
else
    echo -e "${_RED}  ${#FAILED[@]} check(s) failed.${_RST}  See $LOG_FILE for details."
    if [[ $STATUS_ONLY -eq 1 ]]; then
        echo -e "  Fix the items above, then re-run: bash scripts/install.sh"
    fi
    echo ""
    exit 1
fi
