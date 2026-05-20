"""HuggingFace cache janitor for ephemeral expedition runs.

Tracks which model repos were pre-existing before the run, decides whether
a result earns gold-star preservation, and evicts net-new downloads when
the model doesn't qualify.

Public API:
    snapshot_preexisting() -> frozenset[str]
    is_gold_star(result: ScoreResult) -> bool
    maybe_evict(model_id, result, preexisting, evict_failures=False) -> tuple[bool, int]
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import scan_cache_dir

log = logging.getLogger(__name__)


def snapshot_preexisting() -> frozenset[str]:
    """Return the set of HF model repo IDs already in cache before this run.

    Called once at worker startup. Returns an empty frozenset on any error so
    the expedition continues — the conservative path is to treat everything as
    pre-existing (nothing gets evicted) rather than aborting the run.
    """
    try:
        info = scan_cache_dir()
        return frozenset(repo.repo_id for repo in info.repos)
    except Exception as exc:
        log.warning("cache_janitor: snapshot failed (%s); treating all as pre-existing", exc)
        return frozenset()


def is_gold_star(result: "ScoreResult") -> bool:
    """Return True if a successful result earns gold-star preservation.

    Gold star = pts > 0 AND (rarity rare/legendary OR first-ever compile).
    Failures (pts <= 0) are never gold star even if rarity is legendary.
    """
    if result.pts <= 0:
        return False
    return result.rarity in ("rare", "legendary") or result.is_first_ever


def _hf_repo_dir(model_id: str) -> Path:
    """Map a HuggingFace model_id to its local cache directory path.

    Uses HF_HUB_CACHE (from huggingface_hub.constants) so that the
    HF_HOME and HUGGINGFACE_HUB_CACHE environment overrides are respected,
    rather than always resolving to ~/.cache/huggingface/hub/.
    """
    from huggingface_hub.constants import HF_HUB_CACHE
    safe = model_id.replace("/", "--")
    return Path(HF_HUB_CACHE) / f"models--{safe}"


def maybe_evict(
    model_id: str,
    result,
    preexisting: frozenset[str],
    evict_failures: bool = False,
) -> tuple[bool, int]:
    """Evict model weights from HF cache if appropriate.

    Only acts on net-new downloads (not in preexisting). Returns
    (evicted, bytes_freed). Never raises — logs and returns (False, 0) on
    any error so the expedition is never aborted by cleanup failures.

    Args:
        model_id:       HuggingFace model identifier, e.g. "openai-community/gpt2".
        result:         ScoreResult from scorer.compute_score().
        preexisting:    frozenset of repo_id strings present before this run.
        evict_failures: If True, also evict weights for failed models.
    """
    if model_id in preexisting:
        return False, 0

    repo_dir = _hf_repo_dir(model_id)
    if not repo_dir.exists():
        return False, 0

    success = result.pts > 0
    if success:
        if is_gold_star(result):
            return False, 0
    else:
        if not evict_failures:
            return False, 0

    bytes_freed = 0
    try:
        info = scan_cache_dir()
        repo_info = next((r for r in info.repos if r.repo_id == model_id), None)
        if repo_info:
            bytes_freed = repo_info.size_on_disk
    except Exception as exc:
        log.debug("cache_janitor: could not measure size for %s: %s", model_id, exc)

    try:
        shutil.rmtree(repo_dir)
        return True, bytes_freed
    except Exception as exc:
        log.warning("cache_janitor: eviction failed for %s: %s", model_id, exc)
        return False, 0
