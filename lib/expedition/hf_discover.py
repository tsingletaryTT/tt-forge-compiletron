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
    "image-text-to-text":           ("AutoModelForVision2Seq",            "image"),
}

# Substrings that hint at large Mixture-of-Experts architectures — these
# benefit from (and may require) multi-chip mesh configurations.
_LARGE_MOE_PATTERNS = ["deepseek", "mixtral", "qwen", "kimi"]

# Model ID substrings that identify formats forge cannot compile.  These are
# binary-quantized or runtime-specific formats that transformers cannot load
# via AutoModel.from_pretrained(), so attempting them wastes download bandwidth.
_UNSUPPORTED_FORMAT_PATTERNS = {
    "-gguf",       # llama.cpp GGUF
    ".gguf",       # llama.cpp GGUF (dot variant)
    "_gguf",       # llama.cpp GGUF (underscore variant)
    "-ggml",       # older llama.cpp GGML
    "-exl2",       # ExLlamaV2
    "-llamafile",  # Mozilla llamafile
    "-mlx",        # Apple MLX (Silicon-specific)
    "_mlx",        # Apple MLX (underscore variant)
}

# config.model_type values that cannot be compiled by forge without special
# dependencies (mamba-ssm, custom CUDA kernels) or that transformers does not
# recognise at all.  Checking this at discovery time saves the full download +
# load cycle for models that would always fail in the worker.
#
# Grouped by root cause to make future maintenance easier.
_UNSUPPORTED_MODEL_TYPES: frozenset[str] = frozenset({
    # ── Requires mamba-ssm + causal-conv1d ─────────────────────────────────
    # These models call selective_scan / causal_conv1d ops from the mamba-ssm
    # package, which is not installed in the Tenstorrent forge environment.
    "mamba",            # original Mamba SSM (Gu & Dao 2023)
    "mamba2",           # Mamba-2 (improved SSM)
    "falcon_mamba",     # TII Falcon Mamba variant
    "jamba",            # AI21 hybrid Attention+Mamba
    "zamba",            # Zamba hybrid
    "zamba2",           # Zamba-2
    "bamba",            # IBM Bamba hybrid
    "samba",            # Samba SSM variant
    "ssm_nemotron",     # NVIDIA Nemotron SSM (e.g. Nemotron-3-Nano)

    # ── Pure RNN / linear-recurrent ─────────────────────────────────────────
    # No self-attention at all; require custom recurrent kernels that
    # TorchScript / forge cannot trace.
    "rwkv",             # RWKV v4 (pure RNN)
    "rwkv6",            # RWKV v6 / RWKV-6
    "rwkv6_attn",       # RWKV-6 hybrid attention variant
    "hgrn",             # Hierarchical Gated Recurrent Network
    "hgrn2",            # HGRNv2
    "retnet",           # RetNet (Microsoft)
    "xlstm",            # xLSTM (extended LSTM)
    "hawk",             # DeepMind Hawk
    "griffin",          # DeepMind Griffin
    "recurrent_gemma",  # Google recurrent Gemma
    "mega",             # MEGA moving-average gated attention (custom scan)

    # ── Requires custom CUDA kernels not available in forge ──────────────────
    "megalodon",        # Meta Megalodon (chunked CEMA kernel)

    # ── Not in the current transformers model registry ───────────────────────
    # These produce ValueError: "Unrecognized model in config: model_type=…".
    # Listed explicitly so discovery skips the download entirely.
    # Add new entries here whenever a worker failure surfaces a new custom type.
    "qwen3_attnres",    # custom Qwen3 attention-residual experiment
    "dwa",              # Dynamic Weight Adjustment (unknown custom code)
    "bitnet",           # BitNet 1.58-bit (requires custom int1 kernels)
    "bitnet_b158",      # BitNet variant
    "glm4_moe",         # GLM-4 MoE variant — not in current transformers registry
    "nemotron_labs_diffusion",  # NVIDIA Nemotron Labs diffusion — custom config class
})

# Parameter count threshold (in billions) above which a model is flagged for
# multi-chip mesh placement regardless of architecture name.
_LARGE_PARAM_THRESHOLD_B = 40

# HuggingFace tags that mark a model as hobbyist-only with no compelling
# hardware use-case story.  Any model carrying one of these tags is skipped
# during frontier discovery.  Keep this list narrow — broad terms like
# "assistant" or "chat" catch legitimate models.
_BLOCKED_TAGS: frozenset[str] = frozenset({
    "sillytavern",   # SillyTavern character roleplay UI — no research value
    "DarkIdol",      # specific roleplay character-card series
    "OpenClaw",      # roleplay claw-card series
})

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
    library:      HuggingFace library tag (e.g. "pytorch", "jax"). Used by
                  router.py to route jax/flax models to the XLA backend.
    model_type:   Value of config.json "model_type" (e.g. "bert", "gpt2").
                  Used by router.py for architecture-affinity XLA routing.
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
    library: str = ""
    model_type: str = ""


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

    # Extract library and model_type for backend routing.
    library = getattr(hf_model, "library_name", "") or ""
    config = getattr(hf_model, "config", None)
    model_type = (config.get("model_type") or "") if isinstance(config, dict) else ""

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
        library=library,
        model_type=model_type,
    )


def discover_frontier(
    compiled_ids: set[str],
    known_model_ids: set[str],
    limit: int = 5000,
    min_downloads: int = 0,
    min_likes: int = 0,
    max_dl_like_ratio: int = 0,
    max_params_b: float = 0.0,
    skip_gated: bool = True,
    library: str | None = "pytorch",
    max_age_days: int = 180,
) -> list[FrontierModel]:
    """Query HuggingFace for recent, well-regarded models not yet in the bestiary.

    Strategy: sort by ``createdAt`` (newest first) with ``filter='pytorch'`` and
    a large page limit.  Filtering by library tag narrows the pool to pytorch-
    compatible models and extends the time window covered: without a filter, the
    5000 most-recently-created models are all from today (zero downloads); with
    ``filter='pytorch'`` the same 5000 slots cover ~100+ days back in time and
    include real models that have accumulated downloads and likes.

    Sorting by ``downloads`` without a recency filter yields only old models
    (BERT, GPT-2) that are all filtered out by ``max_age_days``, so ``createdAt``
    is the correct primary sort.

    Filter passes applied in order:

    1. ``pipeline_tag`` must be in ``_SUPPORTED_TAGS``.
    2. Model ID not a known-unsupported binary format (GGUF, GGML, etc.).
    3. ``config.model_type`` must be present when config data is returned —
       models without ``model_type`` reliably fail with "Unrecognized model".
    4. ``config.model_type`` not in ``_UNSUPPORTED_MODEL_TYPES`` — saves the
       full download + load cycle for architectures that always fail in the
       worker (mamba-ssm dependent, linear-recurrent, unknown custom types).
    5. Model ID not in ``compiled_ids`` or ``known_model_ids``.
    6. Not a duplicate ID (HF API can return the same repo twice).
    7. Quality bar: ``downloads >= min_downloads``, ``likes >= min_likes``.
    8. Engagement ratio: when ``max_dl_like_ratio > 0``, models where
       ``downloads / likes > max_dl_like_ratio`` are rejected.  Bots inflate
       raw download counts without generating ♥ clicks, so a lopsided ratio
       is the strongest single signal of artificial traffic.  Only applied
       when ``likes > 0`` (the ``min_likes`` floor handles the zero-likes case).
    9. Gated models skipped when ``skip_gated=True`` (can't be downloaded
       without explicit HuggingFace approval).
    10. Size cap: when ``max_params_b > 0``, models whose safetensors metadata
       reports more than that many billion parameters are skipped.  Models
       with no safetensors metadata are passed through (size unknown).

    Parameters
    ----------
    compiled_ids:      Already-compiled model IDs — skip these.
    known_model_ids:   Forge-models library IDs — handled elsewhere, skip.
    limit:             Maximum number of HF API results to fetch.
    min_downloads:     Quality floor — models below this are experiments/noise.
    min_likes:         Community-reputation floor.  Likes require a human click
                       and are much harder to inflate artificially than downloads.
    max_dl_like_ratio: Engagement cap — reject models where
                       ``downloads / likes > max_dl_like_ratio``.  A ratio of
                       300 means a model with 30,000 downloads must have ≥ 100
                       likes to qualify.  Set to 0 to disable (no ratio check).
    max_params_b:      Size ceiling in billions of parameters (0 = no limit).
    skip_gated:        Skip gated repos that require HF access approval.
    library:           HuggingFace library tag to filter by (e.g. "pytorch",
                       "jax", "flax").  Pass ``None`` to discover models from
                       all libraries — useful for XLA/auto modes that want
                       Flax-native models alongside PyTorch ones.
                       Defaults to "pytorch" for backwards compatibility.
    max_age_days:      Only include models created within this many days.
                       Filters out legacy models while keeping genuinely recent
                       work.  0 = no age limit.  Default 180 (6 months).

    Returns an empty list if ``huggingface_hub`` is unavailable or the API
    call fails — callers should treat that as "no new discoveries this tick".
    """
    if HfApi is None:
        return []

    api = HfApi()
    try:
        api_kwargs = dict(
            sort="createdAt",
            direction=-1,   # descending → newest first
            limit=limit,
            # expand overrides the default field set — list every field we read.
            # Omitting any field here leaves it as None in the response.
            expand=[
                "config",        # config.json dict for model_type filter
                "pipeline_tag",  # task type for _SUPPORTED_TAGS check
                "downloads",     # quality floor
                "likes",         # community-reputation floor
                "gated",         # access-control flag
                "disabled",      # archived/deleted flag
                "safetensors",   # weight-file metadata for size cap
                "createdAt",     # upload timestamp — primary sort key
            ],
        )
        # Only pass filter when a library is specified — omitting it discovers all libraries.
        # Use falsy check so empty string "" is treated the same as None (no filter).
        if library:
            api_kwargs["filter"] = library
        hf_models = api.list_models(**api_kwargs)
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
        # Skip binary-quantized / runtime-specific formats forge can't compile.
        model_id_lower = m.id.lower()
        if any(pat in model_id_lower for pat in _UNSUPPORTED_FORMAT_PATTERNS):
            _log.debug("skipped_unsupported_format model=%s", m.id)
            continue
        # Skip models whose config.json has no model_type — these produce
        # "Unrecognized model" errors and account for ~70 % of run failures.
        # We only filter when config was returned (not None) to avoid rejecting
        # models where the expand call simply didn't include config data.
        config = getattr(m, "config", None)
        if isinstance(config, dict):
            model_type = config.get("model_type") or ""
            if not model_type:
                _log.debug("skipped_no_model_type model=%s", m.id)
                continue
            # Skip architectures whose model_type is in our blocklist — these
            # require special packages (mamba-ssm, custom CUDA kernels) or are
            # not in the transformers registry and would always fail in the worker.
            if model_type.lower() in _UNSUPPORTED_MODEL_TYPES:
                _log.debug("skipped_unsupported_model_type model=%s type=%s",
                           m.id, model_type)
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
        # Engagement ratio guard — bots inflate downloads without generating likes.
        # A healthy model has at most max_dl_like_ratio downloads per like.
        # Only checked when the ratio cap is enabled and likes > 0 (the min_likes
        # floor already handles the zero-likes case).
        if max_dl_like_ratio > 0 and lk > 0 and dl / lk > max_dl_like_ratio:
            _log.debug("skipped_bot_ratio model=%s downloads=%d likes=%d ratio=%.0f",
                       m.id, dl, lk, dl / lk)
            continue
        # Reject models whose tags mark them as hobbyist-only (e.g. SillyTavern
        # character roleplay).  Checked against the full tag list, not just
        # pipeline_tag, because these markers appear in the card metadata tags.
        model_tags = {t.lower() for t in (getattr(m, "tags", None) or [])}
        if model_tags & {t.lower() for t in _BLOCKED_TAGS}:
            _log.debug("skipped_blocked_tag model=%s", m.id)
            continue
        # Gated models require an explicit HF access grant — skip by default.
        if skip_gated and getattr(m, "gated", None):
            _log.debug("skipped_gated model=%s", m.id)
            continue
        # Skip disabled repos (archived / deleted but still indexed).
        if getattr(m, "disabled", None):
            continue
        # Recency gate: respect max_age_days if set, and drop old low-traction models.
        created_at = getattr(m, "created_at", None)
        if created_at is not None:
            age_days = (datetime.now(timezone.utc) - created_at).days
            if max_age_days > 0 and age_days > max_age_days:
                _log.debug("skipped_too_old model=%s age_days=%d", m.id, age_days)
                continue
            if age_days > _OLD_MODEL_AGE_DAYS and dl < _OLD_MODEL_MIN_DOWNLOADS:
                _log.debug("skipped_old_low_traction model=%s age_days=%d downloads=%d",
                           m.id, age_days, dl)
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


# Models older than this many days need at least OLD_MODEL_MIN_DOWNLOADS to
# qualify via the author supplement.  Recent uploads from proven authors are
# always included; old experiments with tiny traction are skipped so we don't
# prove out the past when a newer version of the same work likely exists.
_OLD_MODEL_AGE_DAYS = 365
_OLD_MODEL_MIN_DOWNLOADS = 10_000


def discover_from_authors(
    authors: list[str],
    compiled_ids: set[str],
    known_model_ids: set[str],
    max_per_author: int = 15,
    skip_gated: bool = True,
    library: str | None = "pytorch",
) -> list[FrontierModel]:
    """Return uncompiled models from authors whose models have already compiled.

    When the main frontier scan yields few candidates (sparse recent window),
    supplementing with proven-author models keeps the queue full with targets
    that are statistically likely to compile — same author usually means same
    base architecture and same upload patterns.

    Applies the same filters as discover_frontier (tag, format, model_type,
    gated, disabled) plus a recency gate: models older than _OLD_MODEL_AGE_DAYS
    must have at least _OLD_MODEL_MIN_DOWNLOADS to qualify.  This prevents old
    low-traction experiments from clogging the queue when a newer version of the
    same work likely exists on HuggingFace.

    Args:
        authors:         HuggingFace author names (namespace before the /).
        compiled_ids:    Model IDs already in the bestiary — skip these.
        known_model_ids: Forge-models library IDs — handled elsewhere, skip.
        max_per_author:  Maximum HF API results to fetch per author.
        skip_gated:      Skip gated repos requiring HF access approval.
        library:         HuggingFace library tag to filter by (e.g. "pytorch",
                         "jax", "flax").  Pass ``None`` to discover models from
                         all libraries — useful for XLA/auto modes that want
                         Flax-native models.  Defaults to "pytorch" for
                         backwards compatibility.
    """
    if HfApi is None or not authors:
        return []

    api = HfApi()
    results: list[FrontierModel] = []
    # Unified exclusion set so we never return already-known models.
    excluded = set(compiled_ids) | set(known_model_ids)
    seen_ids: set[str] = set()
    now = datetime.now(timezone.utc)

    for author in authors:
        try:
            author_kwargs = dict(
                author=author,
                sort="createdAt",
                direction=-1,
                limit=max_per_author,
                expand=["config", "pipeline_tag", "downloads", "likes",
                        "gated", "disabled", "safetensors", "createdAt"],
            )
            # Only pass filter when a library is specified — omitting it discovers all libraries.
            # Use falsy check so empty string "" is treated the same as None (no filter).
            if library:
                author_kwargs["filter"] = library
            hf_models = list(api.list_models(**author_kwargs))
        except Exception:
            continue

        for m in hf_models:
            if m.id in excluded or m.id in seen_ids:
                continue
            tag = getattr(m, "pipeline_tag", None)
            if not tag or tag not in _SUPPORTED_TAGS:
                continue
            if any(pat in m.id.lower() for pat in _UNSUPPORTED_FORMAT_PATTERNS):
                continue
            config = getattr(m, "config", None)
            if isinstance(config, dict):
                mt = (config.get("model_type") or "").lower()
                if not mt or mt in _UNSUPPORTED_MODEL_TYPES:
                    continue
            if skip_gated and getattr(m, "gated", None):
                continue
            if getattr(m, "disabled", None):
                continue
            # Recency gate: old models with little traction are likely superseded.
            # Prefer trying the newer version from the same author rather than
            # burning bandwidth on a low-download experiment from a year+ ago.
            created_at = getattr(m, "created_at", None)
            if created_at is not None:
                age_days = (now - created_at).days
                dl = getattr(m, "downloads", 0) or 0
                if age_days > _OLD_MODEL_AGE_DAYS and dl < _OLD_MODEL_MIN_DOWNLOADS:
                    _log.debug("skipped_old_low_traction model=%s age_days=%d downloads=%d",
                               m.id, age_days, dl)
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
        # trust_remote_code: safe no-op for standard models; required for repos
        # that ship custom modeling code (detected by `auto_map` in config.json).
        model = AutoClass.from_pretrained(model_id, trust_remote_code=True)
        # return_dict=False makes every transformer return a plain tuple instead
        # of a ModelOutput dataclass — required for TorchScript tracing in forge.
        model.config.return_dict = False
        return model

    # Annotate the closure so the harness can introspect without calling it.
    loader.__name__ = f"load_{model_id.replace('/', '_')}"
    loader._input_type = input_type
    loader._model_id = model_id

    # Vision-language models (image-text-to-text) need special handling:
    # 1. The loader returns a _VLForgeWrapper that maps (input_ids, pixel_values)
    #    positionally to kwargs, strips non-tensor outputs (DynamicCache), and
    #    bakes image_sizes as a Python constant so forge never traces it.
    # 2. image_sizes MUST NOT be a forge-traced tensor: the model uses it for
    #    control flow (round(size/patch_size)) which raises TypeError on fake
    #    tensors.  We compute it once from the probe image and capture it in the
    #    wrapper closure so it's a real Python value during forge tracing.
    # 3. _load_inputs uses the chat template so image placeholder tokens appear
    #    in input_ids; without them forward() raises "image tokens do not match".
    if tag == "image-text-to-text":
        _vl_model_id = model_id   # capture for closures below
        _vl_cache: dict = {}      # shared between loader() and _load_inputs_vl()

        _original_loader = loader

        def loader():
            import torch
            import torch.nn as _nn
            import transformers
            from PIL import Image as _Image
            import numpy as _np

            inner = _original_loader()   # load the raw VL model

            # Build probe inputs once; cache for _load_inputs_vl() reuse.
            proc = transformers.AutoProcessor.from_pretrained(
                _vl_model_id, trust_remote_code=True
            )
            pil_img = _Image.fromarray(
                (_np.ones((224, 224, 3)) * 255).astype(_np.uint8)
            )
            conversation = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ]}]
            prompt = proc.apply_chat_template(conversation, add_generation_prompt=True)
            probe = proc(images=pil_img, text=prompt, return_tensors="pt")

            # Cache probe tensors so _load_inputs_vl doesn't re-run the processor.
            _vl_cache["input_ids"]    = probe["input_ids"]
            _vl_cache["pixel_values"] = probe["pixel_values"]

            # image_sizes captured as a concrete Python tensor — NOT forwarded as
            # a forge-traced input.  The model's internal round()/int() calls on
            # image dimensions work correctly on real values but crash on fake tensors.
            _image_sizes_const = probe["image_sizes"]

            class _VLForgeWrapper(_nn.Module):
                """Wraps a vision-language model for forge compilation.

                Forward takes only (input_ids, pixel_values) so forge never
                traces image_sizes.  image_sizes is baked as a Python constant
                in the closure, keeping round()/int() calls concrete.
                """
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids, pixel_values):
                    out = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        image_sizes=_image_sizes_const,   # constant, not traced
                        use_cache=False,
                    )
                    # return_dict=False → tuple; keep only tensor elements.
                    if isinstance(out, (tuple, list)):
                        tensors = [o for o in out if isinstance(o, torch.Tensor)]
                        return tensors[0] if len(tensors) == 1 else tuple(tensors)
                    return out

            wrapped = _VLForgeWrapper(inner)
            wrapped.config = inner.config
            return wrapped

        loader.__name__ = f"load_{_vl_model_id.replace('/', '_')}"
        loader._input_type = input_type
        loader._model_id = _vl_model_id

        def _load_inputs_vl():
            # Reuse probe tensors computed in loader() if available.
            if _vl_cache:
                return {
                    "input_ids":    _vl_cache["input_ids"],
                    "pixel_values": _vl_cache["pixel_values"],
                }
            # Fallback: loader() wasn't called yet — compute fresh.
            import transformers
            from PIL import Image as _Image
            import numpy as _np

            proc = transformers.AutoProcessor.from_pretrained(
                _vl_model_id, trust_remote_code=True
            )
            pil_img = _Image.fromarray(
                (_np.ones((224, 224, 3)) * 255).astype(_np.uint8)
            )
            conversation = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ]}]
            prompt = proc.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = proc(images=pil_img, text=prompt, return_tensors="pt")
            return {
                "input_ids":    inputs["input_ids"],
                "pixel_values": inputs["pixel_values"],
            }

        loader._load_inputs = _load_inputs_vl

    return loader
