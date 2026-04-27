from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Rarity(str, Enum):
    FAMILIAR = "familiar"
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    LEGENDARY = "legendary"


class Newness(str, Enum):
    ZERO_DAY = "zero_day"
    HOT = "hot"
    FRESH = "fresh"
    RECENT = "recent"
    ESTABLISHED = "established"
    FAMILIAR = "familiar"


@dataclass
class ScoreResult:
    pts: int
    is_first_ever: bool
    rarity: Rarity
    newness: Newness
    streak_at_score: int
    breakdown: dict = field(default_factory=dict)


_RARITY_THRESHOLDS = [
    (10_000_000, Rarity.LEGENDARY),
    (1_000_000,  Rarity.RARE),
    (100_000,    Rarity.UNCOMMON),
]

_NEWNESS_THRESHOLDS_DAYS = [
    (1,   Newness.ZERO_DAY),
    (7,   Newness.HOT),
    (30,  Newness.FRESH),
    (90,  Newness.RECENT),
]

_RARITY_MULT = {
    Rarity.FAMILIAR:  1.0,
    Rarity.COMMON:    1.0,
    Rarity.UNCOMMON:  1.5,
    Rarity.RARE:      2.0,
    Rarity.LEGENDARY: 4.0,
}

_NEWNESS_MULT = {
    Newness.ZERO_DAY:    5.0,
    Newness.HOT:         3.0,
    Newness.FRESH:       2.0,
    Newness.RECENT:      1.5,
    Newness.ESTABLISHED: 1.0,
    Newness.FAMILIAR:    1.0,
}


def compute_rarity(hf_downloads: int | None) -> Rarity:
    """Map a HuggingFace download count to a Rarity tier.

    None means the model is not tracked on HF, so it's treated as FAMILIAR
    (locally known, low-profile). Otherwise we walk thresholds from highest
    to lowest and return the first that matches.
    """
    if hf_downloads is None:
        return Rarity.FAMILIAR
    for threshold, rarity in _RARITY_THRESHOLDS:
        if hf_downloads >= threshold:
            return rarity
    return Rarity.COMMON


def compute_newness(hf_created_at: str | None, is_first_ever: bool) -> Newness:
    """Determine how "new" a compilation event is, based on model age and
    whether this is the first-ever successful compile.

    Rules:
    - If not the first compile for this model, it's ESTABLISHED regardless of age.
    - If no creation date is available, fall back to FAMILIAR.
    - Otherwise bucket the model age in days against _NEWNESS_THRESHOLDS_DAYS.
    """
    if not is_first_ever:
        return Newness.ESTABLISHED
    if hf_created_at is None:
        return Newness.FAMILIAR
    try:
        created = datetime.fromisoformat(hf_created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
    except (ValueError, TypeError):
        return Newness.ESTABLISHED
    for max_days, newness in _NEWNESS_THRESHOLDS_DAYS:
        if age_days < max_days:
            return newness
    return Newness.ESTABLISHED


def compute_score(
    success: bool,
    is_first_ever: bool,
    rarity: Rarity,
    newness: Newness,
    streak: int,
    mesh_chips: int = 1,
) -> ScoreResult:
    """Compute the expedition score for a single compile attempt.

    Formula (success path):
        pts = int((base + first_ever_bonus) * rarity_mult * newness_mult * streak_mult) + mesh_bonus

    Parameters
    ----------
    success:       Whether compilation succeeded.
    is_first_ever: True if this is the first successful compile of this model ever.
    rarity:        Rarity tier (drives a multiplier on the combined base+bonus).
    newness:       How recently the model appeared on HF (multiplier, only on first-ever).
    streak:        Consecutive successes before this one (adds 10% per streak, capped at 2x).
    mesh_chips:    Number of TT chips in the mesh (4+ earns a flat mesh bonus).

    Returns a ScoreResult with pts and a full breakdown dict for audit/display.
    """
    if not success:
        return ScoreResult(
            pts=-10, is_first_ever=is_first_ever, rarity=rarity, newness=newness,
            streak_at_score=streak, breakdown={"failure": -10},
        )

    base = 50
    first_ever_bonus = 100 if is_first_ever else 0
    rarity_mult = _RARITY_MULT[rarity]
    newness_mult = _NEWNESS_MULT[newness] if is_first_ever else 1.0
    # Streak: +10% per consecutive success, hard-capped at 2x.
    streak_mult = min(1.0 + streak * 0.1, 2.0)

    # Flat mesh bonus for multi-chip runs — "galaxy brain" tier at 32 chips.
    mesh_bonus = 0
    if mesh_chips >= 32:
        mesh_bonus = 200
    elif mesh_chips >= 4:
        mesh_bonus = 50

    pts = int((base + first_ever_bonus) * rarity_mult * newness_mult * streak_mult) + mesh_bonus

    return ScoreResult(
        pts=pts,
        is_first_ever=is_first_ever,
        rarity=rarity,
        newness=newness,
        streak_at_score=streak,
        breakdown={
            "base": base,
            "first_ever_bonus": first_ever_bonus,
            "rarity_mult": rarity_mult,
            "newness_mult": newness_mult,
            "streak_mult": streak_mult,
            "mesh_bonus": mesh_bonus,
        },
    )
