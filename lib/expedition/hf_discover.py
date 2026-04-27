# lib/expedition/hf_discover.py
"""HuggingFace frontier discovery module.

Queries the HuggingFace Hub for the newest models (sorted by upload date,
newest first) and filters them against already-compiled and known-library IDs
to surface zero-day compilation targets for Expedition Mode.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable
from lib.expedition.scorer import (
    Rarity, Newness, compute_rarity, compute_newness,
)

try:
    from huggingface_hub import HfApi
except ImportError:
    # Defensive guard — huggingface_hub should always be present in the
    # Tenstorrent venv, but we degrade gracefully rather than crashing on import.
    HfApi = None


# Pipeline tags we actually have AutoModel classes for and can attempt to
# compile.  Anything outside this set is silently skipped during discovery.
_SUPPORTED_TAGS = {
    "text-generation", "text2text-generation", "fill-mask",
    "question-answering", "image-classification", "object-detection",
    "semantic-segmentation", "image-segmentation", "depth-estimation",
    "automatic-speech-recognition", "audio-classification",
    "image-to-text", "visual-question-answering", "image-captioning",
    "text-to-speech", "text-to-image",
}

# Maps pipeline_tag → (transformers AutoModel class name, dummy input type).
# The input type string is attached to the loader closure so callers can
# construct an appropriate dummy tensor without inspecting the model.
_TAG_TO_AUTO = {
    "text-generation":              ("AutoModelForCausalLM",              "text"),
    "text2text-generation":         ("AutoModelForSeq2SeqLM",             "text"),
    "fill-mask":                    ("AutoModelForMaskedLM",              "text"),
    "question-answering":           ("AutoModelForQuestionAnswering",     "text"),
    "image-classification":         ("AutoModelForImageClassification",   "image"),
    "object-detection":             ("AutoModelForObjectDetection",       "image"),
    "semantic-segmentation":        ("AutoModelForSemanticSegmentation",  "image"),
    "image-segmentation":           ("AutoModelForImageSegmentation",     "image"),
    "depth-estimation":             ("AutoModelForDepthEstimation",       "image"),
    "automatic-speech-recognition": ("AutoModelForSpeechSeq2Seq",        "audio"),
    "audio-classification":         ("AutoModelForAudioClassification",   "audio"),
    "image-to-text":                ("AutoModelForVision2Seq",            "image"),
    "visual-question-answering":    ("AutoModelForVision2Seq",            "image"),
    "image-captioning":             ("AutoModelForVision2Seq",            "image"),
}

# Substrings that hint at large Mixture-of-Experts architectures — these
# benefit from (and may require) multi-chip mesh configurations.
_LARGE_MOE_PATTERNS = ["deepseek", "mixtral", "qwen", "kimi"]

# Parameter count threshold (in billions) above which a model is flagged for
# multi-chip mesh placement regardless of architecture name.
_LARGE_PARAM_THRESHOLD_B = 40

_log = logging.getLogger(__name__)

# Matches size tokens like "7B", "1.3B", "0.6b", "70b" in a model ID.
# Used as a fallback when safetensors metadata is absent (e.g. GGUF models).
_PARAM_RE = re.compile(r"(?<![.\d])(\d+(?:\.\d+)?)[Bb](?!\w)")


def _parse_params_from_name(model_id: str) -> float:
    """Extract the largest N from any 'NB' token in the model ID, or 0.0."""
    hits = [float(m.group(1)) for m in _PARAM_RE.finditer(model_id)]
    return max(hits) if hits else 0.0


@dataclass
class FrontierModel:
    """A model candidate surfaced from the HuggingFace frontier.

    Carries everything the scoring and compilation pipeline needs to decide
    whether and how to attempt a compile run.

    Attributes
    ----------
    model_id:     HuggingFace repo ID (e.g. "mistralai/Mistral-7B-v0.1").
    pipeline_tag: Task tag from the HF model card (e.g. "text-generation").
    downloads:    Total all-time download count on HuggingFace.
    likes:        HuggingFace ♥ count — proxy for community reputation.
    params_b:     Approximate parameter count in billions from safetensors
                  metadata; 0.0 when the metadata is absent.
    created_at:   UTC datetime when the repo was first created (may be None).
    rarity:       Rarity tier derived from download count.
    newness:      How recently the model was published.
    mesh_chips:   Recommended chip count; defaults to 1, raised to 4 for
                  large MoE models detected by name heuristic.
    """
    model_id: str
    pipeline_tag: str
    downloads: int
    likes: int
    params_b: float
    created_at: Optional[datetime]
    rarity: Rarity
    newness: Newness
    mesh_chips: int = 1


def _model_to_frontier(hf_model) -> FrontierModel:
    """Convert a raw HuggingFace ModelInfo object into a FrontierModel.

    Computes rarity from the download count and newness from the creation
    timestamp.  We always pass ``is_first_ever=True`` here because frontier
    discovery only surfaces models we have never compiled before — the caller
    filters out already-compiled IDs before we ever see them.

    Also applies a name-based heuristic to bump mesh_chips to 4 for models
    whose IDs contain known large-MoE keywords (deepseek, mixtral, etc.).
    """
    created_at = getattr(hf_model, "created_at", None)
    # Convert datetime → ISO string for compute_newness which expects a string.
    created_str = created_at.isoformat() if created_at else None
    downloads = getattr(hf_model, "downloads", 0) or 0
    likes = getattr(hf_model, "likes", 0) or 0
    rarity = compute_rarity(downloads)
    # Always treat frontier discoveries as first-ever compiles — callers have
    # already excluded anything in compiled_ids.
    newness = compute_newness(created_str, is_first_ever=True)

    # Multi-chip detection: name-based MoE heuristic OR parameter count >40B.
    # Also extract params_b for caller-side size filtering.
    model_id_lower = hf_model.id.lower()
    moe_name_match = any(p in model_id_lower for p in _LARGE_MOE_PATTERNS)
    params_b = 0.0
    try:
        safetensors = getattr(hf_model, "safetensors", None)
        if safetensors is not None:
            total = getattr(safetensors, "total", None)
            if isinstance(total, (int, float)) and total > 0:
                params_b = total / 1e9
    except Exception:
        pass
    # Fall back to name-based parsing for GGUF and other non-safetensors models
    # that embed their size in the model ID (e.g. "64B", "1.3B").
    if params_b == 0.0:
        params_b = _parse_params_from_name(hf_model.id)
    mesh_chips = 4 if (moe_name_match or params_b > _LARGE_PARAM_THRESHOLD_B) else 1

    return FrontierModel(
        model_id=hf_model.id,
        pipeline_tag=hf_model.pipeline_tag or "",
        downloads=downloads,
        likes=likes,
        params_b=params_b,
        created_at=created_at,
        rarity=rarity,
        newness=newness,
        mesh_chips=mesh_chips,
    )


def discover_frontier(
    compiled_ids: set[str],
    known_model_ids: set[str],
    limit: int = 500,
    min_downloads: int = 0,
    min_likes: int = 0,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
) -> list[FrontierModel]:
    """Query HuggingFace for the newest pytorch models and return uncharted ones.

    Sorts by creation date descending ("newest first") to maximise zero-day
    finds.  Filter passes applied in order:

    1. ``pipeline_tag`` must be in ``_SUPPORTED_TAGS``.
    2. Model ID not in ``compiled_ids`` or ``known_model_ids``.
    3. Not a duplicate ID (HF API can return the same repo twice).
    4. Quality bar: ``downloads >= min_downloads``, ``likes >= min_likes``.
    5. Gated models skipped when ``skip_gated=True`` (can't be downloaded
       without explicit HuggingFace approval).
    6. Size cap: when ``max_params_b > 0``, models whose safetensors metadata
       reports more than that many billion parameters are skipped.  Models
       with no safetensors metadata are passed through (size unknown).

    Parameters
    ----------
    compiled_ids:    Already-compiled model IDs — skip these.
    known_model_ids: Forge-models library IDs — handled elsewhere, skip.
    limit:           Maximum number of HF API results to fetch.
    min_downloads:   Quality floor — models below this are experiments/noise.
    min_likes:       Community-reputation floor.
    max_params_b:    Size ceiling in billions of parameters (0 = no limit).
    skip_gated:      Skip gated repos that require HF access approval.

    Returns an empty list if ``huggingface_hub`` is unavailable or the API
    call fails — callers should treat that as "no new discoveries this tick".
    """
    if HfApi is None:
        return []

    api = HfApi()
    try:
        hf_models = api.list_models(
            filter="pytorch",
            sort="createdAt",
            direction=-1,   # descending → newest first
            limit=limit,
        )
    except Exception:
        # Network errors, auth errors, rate-limits — degrade gracefully.
        return []

    results = []
    seen_ids: set[str] = set()
    for m in hf_models:
        # Skip models with no pipeline tag or an unsupported task type.
        tag = getattr(m, "pipeline_tag", None)
        if not tag or tag not in _SUPPORTED_TAGS:
            _log.debug("skipped_unsupported_task model=%s tag=%s", m.id, tag)
            continue
        # Skip models we already know about or have already compiled.
        if m.id in compiled_ids or m.id in known_model_ids:
            continue
        # Deduplicate — HF API can return the same repo twice.
        if m.id in seen_ids:
            continue
        # Quality / reputation bar.
        dl = getattr(m, "downloads", 0) or 0
        lk = getattr(m, "likes", 0) or 0
        if dl < min_downloads:
            _log.debug("skipped_low_downloads model=%s downloads=%d", m.id, dl)
            continue
        if lk < min_likes:
            _log.debug("skipped_low_likes model=%s likes=%d", m.id, lk)
            continue
        # Gated models require an explicit HF access grant — skip by default.
        if skip_gated and getattr(m, "gated", None):
            _log.debug("skipped_gated model=%s", m.id)
            continue
        # Skip disabled repos (archived / deleted but still indexed).
        if getattr(m, "disabled", None):
            continue
        # Size cap: prefer safetensors metadata; fall back to name-based parsing
        # so GGUF and other non-safetensors models don't bypass the filter.
        if max_params_b > 0:
            inferred_params_b = 0.0
            try:
                st = getattr(m, "safetensors", None)
                if st is not None:
                    total_params = getattr(st, "total", None)
                    if isinstance(total_params, (int, float)) and total_params > 0:
                        inferred_params_b = total_params / 1e9
            except Exception:
                pass
            if inferred_params_b == 0.0:
                inferred_params_b = _parse_params_from_name(m.id)
            if inferred_params_b > max_params_b:
                _log.debug("skipped_too_large model=%s params_b=%.1f",
                           m.id, inferred_params_b)
                continue
        seen_ids.add(m.id)
        results.append(_model_to_frontier(m))

    return results


def build_dynamic_loader(model: FrontierModel) -> Optional[Callable]:
    """Build a lazy loader closure for a FrontierModel.

    Returns a zero-argument callable that, when invoked, imports ``transformers``
    and calls ``AutoXxx.from_pretrained(model_id)`` with the appropriate class
    for the model's pipeline tag.  The closure is annotated with two attributes:

    - ``_input_type`` (str): "text", "image", or "audio" — hints to the
      compilation harness which dummy input shape to construct.
    - ``_model_id`` (str): the HuggingFace repo ID, useful for logging.

    Returns ``None`` if the pipeline tag has no known AutoModel mapping (e.g.
    "reinforcement-learning") — the caller should skip compilation in that case.
    """
    tag = model.pipeline_tag
    if tag not in _TAG_TO_AUTO:
        return None

    auto_class_name, input_type = _TAG_TO_AUTO[tag]
    model_id = model.model_id

    def loader():
        """Lazily load the model from HuggingFace pretrained weights."""
        import transformers
        AutoClass = getattr(transformers, auto_class_name, None)
        if AutoClass is None:
            raise ImportError(
                f"transformers.{auto_class_name} not found — "
                f"upgrade transformers to support pipeline tag '{tag}'"
            )
        return AutoClass.from_pretrained(model_id)

    # Annotate the closure so the harness can introspect without calling it.
    loader.__name__ = f"load_{model_id.replace('/', '_')}"
    loader._input_type = input_type
    loader._model_id = model_id
    return loader
