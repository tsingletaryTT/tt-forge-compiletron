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
