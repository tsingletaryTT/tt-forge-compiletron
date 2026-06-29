# Smart Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/install.sh` — a check-first smart installer that detects gotchas, compares versions against TT PyPI, supports `--status` mode, and delegates all pip work to the existing `scripts/setup-venvs.sh`.

**Architecture:** `install.sh` runs all 11 checks unconditionally (no `set -e`), recording outcomes in `PASSED`/`WARNED`/`FAILED` arrays. In default mode it calls `setup-venvs.sh --forge` or `--xla` when a check fails and is fixable. In `--status` mode it never installs anything. A summary table prints at the end. `setup-venvs.sh` is not modified.

**Tech Stack:** Bash 5, Python 3.12 (one-liner inline scripts for version parsing and forge package validation), `tt-smi -s` JSON output, `/proc/sys/vm/nr_hugepages`, TT PyPI simple index.

## Global Constraints

- No right-side border characters in terminal output (project convention — left and bottom bars only)
- `setup-venvs.sh` must not be modified
- `--status` mode: zero installs, zero sudo, exit 1 if any FAIL
- Log verbose output to `/tmp/tt-compiletron-install.log`
- Version comparison failures (network unreachable) go to WARNED, never FAILED
- Stale `/dev/shm` segments: warn + print fix command only — never auto-delete

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/install.sh` | Create | Smart check + summary wrapper (~350 lines) |
| `INSTALL.md` | Modify (top section) | Point to `install.sh` as the recommended entry point |

---

### Task 1: Shell skeleton, flags, and helpers

**Files:**
- Create: `scripts/install.sh`

**Interfaces:**
- Produces: `ok()`, `warn()`, `fail()`, `info()`, `step()`, `hr()`, `pass_s()`, `warn_s()`, `fail_s()` helpers; `PASSED`/`WARNED`/`FAILED` arrays; `STATUS_ONLY`, `SETUP_FORGE`, `SETUP_XLA`, `SKIP_HARDWARE` flag variables; exit-code logic

- [ ] **Step 1: Create the file with shebang, usage comment, and colour helpers**

```bash
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
    echo -e "${_CYN}╠══════════════════════════════════════════════════${_RST}"
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
```

- [ ] **Step 2: Add version-detection helper**

Append to `scripts/install.sh`:

```bash
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
```

- [ ] **Step 3: Verify the file is valid bash (no syntax errors)**

```bash
bash -n scripts/install.sh
```

Expected: no output, exit 0.

- [ ] **Step 4: Commit skeleton**

```bash
git add scripts/install.sh
git commit -m "feat(install): add shell skeleton, flag parsing, and helpers"
```

---

### Task 2: Steps 1–3 — hardware, hugepages, disk space

**Files:**
- Modify: `scripts/install.sh`

**Interfaces:**
- Consumes: `pass_s()`, `warn_s()`, `fail_s()`, `info()`, `step()`, `STATUS_ONLY`, `SKIP_HARDWARE` from Task 1
- Produces: Steps [1]–[3] written; `FORGE_VENV`, `XLA_VENV`, `XLA_PY`, `FORGE_PY` path variables set

- [ ] **Step 1: Append the header banner and steps 1–3 to `scripts/install.sh`**

```bash
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
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/install.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Quick manual smoke test of hardware + hugepages steps**

```bash
bash scripts/install.sh --status --skip-hardware 2>&1 | head -20
```

Expected: `[1] Hardware presence` line with "skipped", `[2] Hugepages` with current count, `[3] Disk` with current free GB.

- [ ] **Step 4: Commit**

```bash
git add scripts/install.sh
git commit -m "feat(install): add steps 1-3 (hardware, hugepages, disk)"
```

---

### Task 3: Steps 4–6 — forge venv, version, and shim

**Files:**
- Modify: `scripts/install.sh`

**Interfaces:**
- Consumes: `FORGE_VENV`, `FORGE_PY`, `STATUS_ONLY`, `SETUP_FORGE`, `_latest_tt_version()`, `_installed_version()`, `_version_behind()`, `pass_s()`, `warn_s()`, `fail_s()`, `info()`, `step()` from Tasks 1–2
- Produces: Steps [4]–[6] written; `_FORGE_OK` variable (1 if venv is healthy, 0 otherwise)

- [ ] **Step 1: Append forge steps to `scripts/install.sh`**

```bash
# ── [4] Forge venv ────────────────────────────────────────────────────────────
_FORGE_OK=0
if [[ $SETUP_FORGE -eq 1 ]]; then
    step 4 "Forge venv  (~/$( realpath --relative-to="$HOME" "$FORGE_VENV" 2>/dev/null || echo tt-forge-venv))"

    _forge_importable=0
    _forge_is_tt=0

    if [[ -x "$FORGE_PY" ]]; then
        # Check import works
        if "$FORGE_PY" -c "import forge" 2>/dev/null; then
            _forge_importable=1
            # Gotcha: PyPI's "forge" package is a Django app.
            # The TT forge package exposes forge.__version__ as a semver string
            # like "0.22.0" that does NOT contain "Django".
            _forge_ver=$("$FORGE_PY" -c "
import forge, sys
v = getattr(forge, '__version__', '')
# Django forge has no __version__ or its module structure is completely different
try:
    import forge.compiled  # TT forge has this submodule; Django forge does not
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
if [[ $SETUP_FORGE -eq 1 ]]; then
    step 6 "Forge-fe shim  (~/tt-forge-fe/env/activate)"

    _SHIM="$HOME/tt-forge-fe/env/activate"
    if [[ ! -f "$_SHIM" ]]; then
        if [[ $STATUS_ONLY -eq 1 ]]; then
            fail_s "Forge-fe shim missing: $_SHIM"
            info "Fix: bash scripts/setup-venvs.sh --forge"
        elif [[ $_FORGE_OK -eq 1 ]]; then
            # setup-venvs.sh already wrote the shim during step 4 fix
            pass_s "Forge-fe shim written by install step above"
        else
            fail_s "Forge-fe shim missing and forge venv install failed — cannot write shim"
        fi
    else
        # Validate it sources cleanly (timeout 5s to avoid hangs)
        if timeout 5 bash -c "source $_SHIM" >> "$LOG_FILE" 2>&1; then
            pass_s "Forge-fe shim: $_SHIM"
        else
            warn_s "Forge-fe shim exists but sourcing it produced errors"
            info "Check: source $_SHIM"
        fi
    fi
fi
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/install.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Smoke test forge steps in status mode**

```bash
bash scripts/install.sh --status --forge --skip-hardware 2>&1
```

Expected: Steps [1]–[6] printed. Step [4] shows `✓` with forge version (since forge is already installed). Step [5] shows installed vs latest. Step [6] shows shim status.

- [ ] **Step 4: Commit**

```bash
git add scripts/install.sh
git commit -m "feat(install): add steps 4-6 (forge venv, version, shim)"
```

---

### Task 4: Steps 7–9 — XLA venv, version, and mesh descriptor

**Files:**
- Modify: `scripts/install.sh`

**Interfaces:**
- Consumes: `XLA_VENV`, `XLA_PY`, `STATUS_ONLY`, `SETUP_XLA`, `_latest_tt_version()`, `_installed_version()`, `_version_behind()`, `pass_s()`, `warn_s()`, `fail_s()`, `info()`, `step()` from Tasks 1–2
- Produces: Steps [7]–[9] written; `_XLA_OK` variable (1 if venv healthy)

- [ ] **Step 1: Append XLA steps to `scripts/install.sh`**

```bash
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

# ── [8] XLA version ───────────────────────────────────────────────────────────
if [[ $SETUP_XLA -eq 1 && $_XLA_OK -eq 1 ]]; then
    step 8 "XLA version  (pjrt-plugin-tt, TT PyPI latest)"

    _pjrt_installed=$(_installed_version "$XLA_PY" "pjrt-plugin-tt")
    # Strip +dev suffix for comparison
    _pjrt_installed_clean="${_pjrt_installed%%+*}"
    _pjrt_latest=$(_latest_tt_version "pjrt-plugin-tt")

    if [[ -z "$_pjrt_latest" ]]; then
        warn_s "XLA version: cannot reach TT PyPI — skipping comparison"
    elif [[ -z "$_pjrt_installed_clean" ]]; then
        warn_s "XLA version: installed version unknown"
    elif _version_behind "$_pjrt_installed_clean" "$_pjrt_latest"; then
        warn_s "XLA version: $_pjrt_installed installed, $_pjrt_latest available"
        info "Upgrade: $XLA_VENV/bin/pip install pjrt-plugin-tt==$_pjrt_latest --extra-index-url $TENSTORRENT_PYPI"
    else
        pass_s "XLA version: $_pjrt_installed (up to date)"
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
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/install.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Smoke test XLA steps in status mode**

```bash
bash scripts/install.sh --status --xla --skip-hardware 2>&1
```

Expected: Steps [1]–[3], [7]–[9] printed. Step [7] shows `✓` with pjrt version. Step [8] shows installed vs latest. Step [9] shows `✓` or fix hint.

- [ ] **Step 4: Commit**

```bash
git add scripts/install.sh
git commit -m "feat(install): add steps 7-9 (XLA venv, version, mesh descriptor)"
```

---

### Task 5: Steps 10–11 — tt-forge-models and stale shm

**Files:**
- Modify: `scripts/install.sh`

**Interfaces:**
- Consumes: `pass_s()`, `warn_s()`, `fail_s()`, `info()`, `step()` from Task 1
- Produces: Steps [10]–[11] written

- [ ] **Step 1: Append steps 10–11 to `scripts/install.sh`**

```bash
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
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/install.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/install.sh
git commit -m "feat(install): add steps 10-11 (tt-forge-models, stale shm)"
```

---

### Task 6: Summary table and exit code

**Files:**
- Modify: `scripts/install.sh`

**Interfaces:**
- Consumes: `PASSED`, `WARNED`, `FAILED` arrays; `hr()`, `hr_mid()`, `hr_end()` from Task 1
- Produces: Summary block and correct exit code at end of `install.sh`

- [ ] **Step 1: Append summary block to `scripts/install.sh`**

```bash
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
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/install.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Full end-to-end smoke test — status mode, both backends**

```bash
bash scripts/install.sh --status 2>&1
echo "exit: $?"
```

Expected: all 11 steps printed, summary table at end with `✓`/`⚠`/`✗` per step, correct exit code (0 if no failures).

- [ ] **Step 4: Test `--forge` scope (only forge steps appear)**

```bash
bash scripts/install.sh --status --forge --skip-hardware 2>&1 | grep "^\[" | head -10
```

Expected: only steps [1]–[6] appear, no [7]–[9].

- [ ] **Step 5: Test `--xla` scope**

```bash
bash scripts/install.sh --status --xla --skip-hardware 2>&1 | grep "^\[" | head -10
```

Expected: steps [1]–[3] and [7]–[11] appear, no [4]–[6].

- [ ] **Step 6: Commit**

```bash
git add scripts/install.sh
git commit -m "feat(install): add summary table and exit code"
```

---

### Task 7: Make executable and update INSTALL.md

**Files:**
- Modify: `scripts/install.sh` (chmod)
- Modify: `INSTALL.md`

**Interfaces:**
- Consumes: completed `scripts/install.sh`
- Produces: `install.sh` executable; INSTALL.md top section updated

- [ ] **Step 1: Make install.sh executable**

```bash
chmod +x scripts/install.sh
```

- [ ] **Step 2: Update the top of INSTALL.md**

Open `INSTALL.md`. Replace the opening paragraph (before the venv table) with:

```markdown
# Installation

The quickest path is the smart installer — it checks your environment,
detects common gotchas (wrong `forge` package, stale shm segments, missing
mesh descriptor), compares installed versions against TT PyPI, and calls
`setup-venvs.sh` to fix anything broken:

```bash
bash scripts/install.sh          # check + fix everything
bash scripts/install.sh --status # check only, no installs
```

For manual step-by-step instructions, source-build options, and
hardware-specific notes, see the sections below.

---

Compiletron needs two Python 3.12 venvs:
```

(Keep everything from the venv table onward unchanged.)

- [ ] **Step 3: Verify INSTALL.md renders correctly**

```bash
head -25 INSTALL.md
```

Expected: smart installer blurb at top, then the original venv table.

- [ ] **Step 4: Full clean smoke test**

```bash
bash scripts/install.sh --status --skip-hardware 2>&1
```

Expected: 11 steps, summary table, correct verdicts, clean exit.

- [ ] **Step 5: Commit everything**

```bash
git add scripts/install.sh INSTALL.md
git commit -m "feat: add scripts/install.sh smart installer

Check-first wrapper around setup-venvs.sh with --status mode,
version comparison against TT PyPI, named gotcha detection
(wrong forge package, dev pjrt build, stale shm), and a
structured pass/warn/fail summary table."
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `--status` check-only mode, exit 1 on FAIL | Tasks 2–6 (STATUS_ONLY flag) |
| `--forge` / `--xla` / `--skip-hardware` / `--help` flags | Task 1 |
| 11 named steps in order | Tasks 2–5 |
| PyPI Django `forge` gotcha detection | Task 3, step [4] |
| `+dev` pjrt build detection | Task 4, step [7] |
| triton/easydel conflict reassurance | Task 4, step [7] |
| Version comparison via TT PyPI simple index | Task 1 (`_latest_tt_version`), Tasks 3–4 |
| Mesh descriptor symlink | Task 4, step [9] |
| Stale shm warn-only, never auto-delete | Task 5, step [11] |
| Summary table with PASSED/WARNED/FAILED | Task 6 |
| Log to `/tmp/tt-compiletron-install.log` | Task 1 (log truncation), Tasks 3–4 (delegated calls) |
| No right-side border characters | Task 1 (`hr()` / `hr_end()`) |
| `setup-venvs.sh` not modified | All tasks — only reads/calls it |
| INSTALL.md updated | Task 7 |

No gaps found.
