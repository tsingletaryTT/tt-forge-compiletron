#!/usr/bin/env bash
# Apply compiletron's local patches to the tt-forge-fe source tree.
#
# Run this after a fresh tt-forge-fe checkout or after a git reset that
# drops the working-tree modifications.  Safe to re-run — git apply uses
# --check first and skips already-applied hunks.
#
# Patches included:
#   forge_pytorch_only.patch — guard TF/Paddle/JAX/Keras imports behind
#       FORGE_PYTORCH_ONLY=1 env var so the forge PyTorch subprocess does
#       not trigger a SIGSEGV from LLVM symbol conflicts between
#       libTTMLIRCompiler.so (--whole-archive LLVM) and TF's own
#       statically-linked LLVM copy.

set -euo pipefail

FORGE_FE="${FORGE_FE:-$HOME/tt-forge-fe}"
PATCHES_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$FORGE_FE/.git" ]]; then
    echo "ERROR: tt-forge-fe not found at $FORGE_FE" >&2
    echo "Set FORGE_FE=/path/to/tt-forge-fe and re-run." >&2
    exit 1
fi

apply_patch() {
    local patch="$1"
    local name; name="$(basename "$patch")"
    echo -n "  $name ... "
    if git -C "$FORGE_FE" apply --check "$patch" 2>/dev/null; then
        git -C "$FORGE_FE" apply "$patch"
        echo "applied"
    else
        # Might already be applied — check with --reverse
        if git -C "$FORGE_FE" apply --check --reverse "$patch" 2>/dev/null; then
            echo "already applied (skipped)"
        else
            echo "CONFLICT — patch does not apply cleanly; forge may have diverged"
            return 1
        fi
    fi
}

echo "Applying compiletron forge patches to: $FORGE_FE"
echo ""
apply_patch "$PATCHES_DIR/forge_pytorch_only.patch"
echo ""
echo "Done."
