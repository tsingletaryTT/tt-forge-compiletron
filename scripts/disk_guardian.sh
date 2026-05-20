#!/bin/bash
# scripts/disk_guardian.sh
# Monitors main drive free space and offloads expedition model weights to
# /mnt/bonus/models when headroom drops below thresholds.
#
# Strategy (two tiers):
#   < WARN_GB  : move compiled expedition weights to bonus (symlink back)
#   < CRIT_GB  : also delete perm-failed expedition weights entirely
#
# Never touches: user LLMs >10 GB, diffusion/image/video models.
# Uses bestiary.json to know which models are compiled vs perm-failed.
#
# Safe to run in a loop — all operations are idempotent.

set -euo pipefail

BESTIARY="/home/ttuser/code/tt-forge-compiletron/data/bestiary.json"
HF_CACHE="$HOME/.cache/huggingface/hub"
BONUS_MODELS="/mnt/bonus/models"
LOG="/tmp/disk_guardian.log"

WARN_GB=200   # start offloading compiled weights below this free space
CRIT_GB=130   # also delete perm-failed weights below this

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

free_gb() {
    df -BG / | awk 'NR==2 {gsub("G",""); print $4}'
}

FREE=$(free_gb)
log "disk check: ${FREE}G free on /"

if (( FREE > WARN_GB )); then
    exit 0
fi

log "WARNING: only ${FREE}G free — running expedition cache offload"

# --- helpers ----------------------------------------------------------------

# Convert HF cache dir name to model ID: models--owner--name -> owner/name
cache_to_id() {
    local d
    d=$(basename "$1")
    d="${d#models--}"
    echo "${d/--/\/}"
}

# Size of a directory in GB (follows symlinks for accurate accounting)
dir_gb() {
    du -s --apparent-size "$1" 2>/dev/null | awk '{printf "%.1f", $1/1048576}'
}

# Move a real cache dir to /mnt/bonus and replace with a symlink.
offload_to_bonus() {
    local src="$1" dst="$2"
    if [[ -L "$src" ]]; then
        return 0  # already a symlink — nothing to do
    fi
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    ln -s "$dst" "$src"
    log "  offloaded $(cache_to_id "$src") -> $dst"
}

# --- pull compiled/perm-failed sets from bestiary ---------------------------

if [[ ! -f "$BESTIARY" ]]; then
    log "bestiary not found at $BESTIARY — skipping"
    exit 0
fi

COMPILED_IDS=$(python3 -c "
import json, sys
b = json.load(open('$BESTIARY'))
for mid in b.get('compiled', {}):
    print(mid)
" 2>/dev/null)

PERM_FAIL_CATS="forge_internal unsupported_arch loader_missing missing_dependency \
unsupported_backend xla_runtime_error api_mismatch shape_mismatch forge_missing_op \
model_access wrong_backend model_bug quantized_format"

PERM_FAILED_IDS=$(python3 -c "
import json, sys
b = json.load(open('$BESTIARY'))
cats = set('$PERM_FAIL_CATS'.split())
for mid, info in b.get('failed', {}).items():
    if info.get('error_category') in cats or info.get('attempts', 0) >= 3:
        print(mid)
" 2>/dev/null)

# --- WARN tier: offload compiled expedition weights to bonus drive -----------

OFFLOADED=0
for cache_dir in "$HF_CACHE"/models--*; do
    [[ -d "$cache_dir" ]] || continue
    [[ -L "$cache_dir" ]] && continue  # already offloaded

    mid=$(cache_to_id "$cache_dir")

    # Skip models larger than 10 GB — those are user's personal LLMs/diffusion.
    gb=$(dir_gb "$cache_dir")
    if (( $(echo "$gb > 10" | bc -l 2>/dev/null || echo 0) )); then
        continue
    fi

    if echo "$COMPILED_IDS" | grep -qxF "$mid"; then
        bonus_dst="$BONUS_MODELS/$(basename "$cache_dir")"
        offload_to_bonus "$cache_dir" "$bonus_dst"
        OFFLOADED=$((OFFLOADED + 1))
    fi
done

[[ $OFFLOADED -gt 0 ]] && log "offloaded $OFFLOADED compiled model(s) to $BONUS_MODELS"

# --- CRIT tier: delete perm-failed weights entirely -------------------------

if (( FREE <= CRIT_GB )); then
    log "CRITICAL: ${FREE}G free — purging perm-failed model weights"
    PURGED=0
    for cache_dir in "$HF_CACHE"/models--*; do
        [[ -e "$cache_dir" ]] || continue
        mid=$(cache_to_id "$cache_dir")
        if echo "$PERM_FAILED_IDS" | grep -qxF "$mid"; then
            if [[ -L "$cache_dir" ]]; then
                target=$(readlink "$cache_dir")
                rm -f "$cache_dir"
                [[ "$target" == /mnt/bonus/* ]] && rm -rf "$target"
                log "  purged (symlink) $mid"
            else
                rm -rf "$cache_dir"
                log "  purged $mid"
            fi
            PURGED=$((PURGED + 1))
        fi
    done
    [[ $PURGED -gt 0 ]] && log "purged $PURGED perm-failed model cache(s)"
fi

FREE_AFTER=$(free_gb)
log "done: ${FREE_AFTER}G free (was ${FREE}G)"
