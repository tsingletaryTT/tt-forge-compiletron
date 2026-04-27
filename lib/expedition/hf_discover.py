# lib/expedition/hf_discover.py
"""HuggingFace frontier discovery module.

Queries the HuggingFace Hub for the newest models (sorted by upload date,
newest first) and filters them against already-compiled and known-library IDs
to surface zero-day compilation targets for Expedition Mode.
"""
from __future__ import annotations
import logging
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
    created_at:   UTC datetime when the repo was first created (may be None).
    rarity:       Rarity tier derived from download count.
    newness:      How recently the model was published.
    mesh_chips:   Recommended chip count; defaults to 1, raised to 4 for
                  large MoE models detected by name heuristic.
    """
    model_id: str
    pipeline_tag: str
    downloads: int
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
    rarity = compute_rarity(downloads)
    # Always treat frontier discoveries as first-ever compiles — callers have
    # already excluded anything in compiled_ids.
    newness = compute_newness(created_str, is_first_ever=True)

    # Multi-chip detection: name-based MoE heuristic OR parameter count >40B.
    model_id_lower = hf_model.id.lower()
    moe_name_match = any(p in model_id_lower for p in _LARGE_MOE_PATTERNS)
    # safetensors metadata carries total parameter count when available.
    large_param = False
    try:
        safetensors = getattr(hf_model, "safetensors", None)
        if safetensors is not None:
            total = getattr(safetensors, "total", None)
            if isinstance(total, (int, float)) and total > 0:
                large_param = total / 1e9 > _LARGE_PARAM_THRESHOLD_B
    except Exception:
        pass
    mesh_chips = 4 if (moe_name_match or large_param) else 1

    return FrontierModel(
        model_id=hf_model.id,
        pipeline_tag=hf_model.pipeline_tag or "",
        downloads=downloads,
        created_at=created_at,
        rarity=rarity,
        newness=newness,
        mesh_chips=mesh_chips,
    )


def discover_frontier(
    compiled_ids: set[str],
    known_model_ids: set[str],
    limit: int = 500,
) -> list[FrontierModel]:
    """Query HuggingFace for the newest pytorch models and return uncharted ones.

    Sorts by creation date descending ("newest first") to maximise zero-day
    finds.  Three filter passes are applied in order:

    1. ``pipeline_tag`` must be in ``_SUPPORTED_TAGS`` — we need an AutoModel
       class for the tag or compilation can't even begin.
    2. Model ID must not be in ``compiled_ids`` — already run, skip.
    3. Model ID must not be in ``known_model_ids`` — part of the canonical
       forge-models library, handled by the standard pipeline, not expedition.

    Parameters
    ----------
    compiled_ids:    Set of model IDs that have already been compiled in this
                     (or any previous) expedition session.
    known_model_ids: Set of model IDs from the forge-models bestiary / known
                     library — these are "tamed" and excluded from frontier.
    limit:           Maximum number of HF results to fetch (API page size).

    Returns an empty list if ``huggingface_hub`` is unavailable or the API
    call fails — callers should treat that as "no new discoveries this tick"
    rather than a hard error.
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
    for m in hf_models:
        # Skip models with no pipeline tag or an unsupported task type.
        tag = getattr(m, "pipeline_tag", None)
        if not tag or tag not in _SUPPORTED_TAGS:
            _log.debug("skipped_unsupported_task model=%s tag=%s", m.id, tag)
            continue
        # Skip models we already know about or have already compiled.
        if m.id in compiled_ids or m.id in known_model_ids:
            continue
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
