#!/usr/bin/env bash
# Apply compiletron's local patches to ~/code/tt-forge-models.
#
# Run this after a `git pull --rebase` on tt-forge-models, or at expedition
# startup via _update_tt_forge_models().  Safe to re-run — patches are
# checked before applying and skipped if already applied.
#
# Patches included:
#   tt_forge_models_gemma3_nnx_mesh.patch — wrap Gemma3ForCausalLM init in a
#       jax.sharding.Mesh context.  nnx sharding annotations require a mesh
#       context even on CPU; without this the bounty_jax loader crashes.

set -euo pipefail

MODELS_DIR="${TT_FORGE_MODELS_DIR:-$HOME/code/tt-forge-models}"
PATCHES_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$MODELS_DIR/.git" ]]; then
    echo "ERROR: tt-forge-models not found at $MODELS_DIR" >&2
    echo "Set TT_FORGE_MODELS_DIR=/path/to/tt-forge-models and re-run." >&2
    exit 1
fi

apply_patch() {
    local patch="$1"
    local name; name="$(basename "$patch")"
    echo -n "  $name ... "
    if git -C "$MODELS_DIR" apply --check "$patch" 2>/dev/null; then
        git -C "$MODELS_DIR" apply "$patch"
        echo "applied"
    else
        if git -C "$MODELS_DIR" apply --check --reverse "$patch" 2>/dev/null; then
            echo "already applied (skipped)"
        else
            echo "CONFLICT — patch does not apply cleanly; tt-forge-models may have diverged"
            return 1
        fi
    fi
}

echo "Applying compiletron tt-forge-models patches to: $MODELS_DIR"
echo ""
apply_patch "$PATCHES_DIR/tt_forge_models_gemma3_nnx_mesh.patch"
echo ""
echo "Done."
