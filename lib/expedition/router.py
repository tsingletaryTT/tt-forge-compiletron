# lib/expedition/router.py
"""Per-model routing decisions: which backend (forge/xla) and how many chips.

This module is stateless — it inspects a queue item dict and a Bestiary snapshot
and returns a DispatchDecision.  It has no side effects and imports nothing from
the TUI or worker layers.
"""
from __future__ import annotations

from dataclasses import dataclass

from lib.expedition.bestiary import Bestiary

# Architectures whose canonical implementation is Flax-native and that have
# already proven out on tt-xla.  Models reporting these model_type values
# are routed to the XLA backend at moderate confidence.
_XLA_AFFINITY_TYPES: frozenset[str] = frozenset({
    "flax_bert",
    "flax_gpt2",
    "flax_roberta",
    "flax_t5",
})

# Error categories that indicate forge cannot handle a model.  Two or more
# failures in these categories redirect future attempts to the XLA backend.
_FORGE_FATAL_CATEGORIES: frozenset[str] = frozenset({
    "forge_missing_op",
    "forge_internal",
})


@dataclass
class DispatchDecision:
    """Routing decision for a single model."""
    backend: str       # "forge" or "xla"
    chips: int         # 1, 2, or 4
    confidence: float  # 0.0–1.0, informational for UI display
    reason: str        # short label: "jax-native", "forge-failure-history", etc.


def route_model(
    item: dict,
    bestiary: Bestiary,
    available_chips: set[int] | None = None,
) -> DispatchDecision:
    """Compute a DispatchDecision for a single queue item.

    Priority order (first match wins):
      1. library == "jax" or "flax" → xla, confidence=0.92, reason="jax-native"
      2. ≥2 forge fatal failures in bestiary → xla, confidence=0.75, reason="forge-failure-history"
      3. model_type in _XLA_AFFINITY_TYPES → xla, confidence=0.68, reason="arch-xla-affinity"
      4. default → forge, confidence=0.60, reason="default"

    Chip count:
      - item["mesh_chips"] is the primary source (set by hf_discover.py heuristics).
      - Falls back to 1 chip if not present.
      - Capped at len(available_chips) if provided.

    Args:
        item:            Queue item dict (model_id, library, mesh_chips, etc.).
        bestiary:        Loaded Bestiary for failure history lookups.
        available_chips: Set of chip IDs in this run (used for cap only).
                         Pass the full chip set (not just free chips) — the TUI
                         enforces free-chip quorum separately.
    """
    model_id   = item.get("model_id", "")
    library    = (item.get("library") or "").lower()
    model_type = (item.get("model_type") or "").lower()

    # ── Backend routing ───────────────────────────────────────────────────────

    # Priority 1: JAX/Flax library tag is definitive.
    if library in ("jax", "flax"):
        backend    = "xla"
        confidence = 0.92
        reason     = "jax-native"

    # Priority 2: Forge has already failed this model with a fundamental error.
    elif _has_forge_fatal_history(model_id, bestiary):
        backend    = "xla"
        confidence = 0.75
        reason     = "forge-failure-history"

    # Priority 3: Architecture is known to work well on XLA.
    elif model_type in _XLA_AFFINITY_TYPES:
        backend    = "xla"
        confidence = 0.68
        reason     = "arch-xla-affinity"

    # Priority 4: Default to forge.
    else:
        backend    = "forge"
        confidence = 0.60
        reason     = "default"

    # ── Chip count ────────────────────────────────────────────────────────────
    chips = int(item.get("mesh_chips", 1)) or 1
    if available_chips is not None and chips > len(available_chips):
        chips = max(1, len(available_chips))

    return DispatchDecision(backend=backend, chips=chips,
                            confidence=confidence, reason=reason)


def _has_forge_fatal_history(model_id: str, bestiary: Bestiary) -> bool:
    """Return True if this model has >=2 forge-fatal failures recorded."""
    entry = bestiary.failed.get(model_id)
    if not entry:
        return False
    if entry.get("error_category") not in _FORGE_FATAL_CATEGORIES:
        return False
    return int(entry.get("attempts", 0)) >= 2
