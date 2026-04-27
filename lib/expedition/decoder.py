# lib/expedition/decoder.py
#
# Converts raw inference output tensors from a compiled model into a
# human-readable artifact string.  Each task family has a dedicated decoder;
# every code path catches exceptions and falls back to raw tensor shape/range
# info so the caller is guaranteed to receive a non-empty string.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FrontierModelInfo:
    """Lightweight description of the model being decoded."""
    name: str
    task: str
    source: str = "huggingface"


# ---------------------------------------------------------------------------
# Task-string → internal family mapping
# Covers both HuggingFace pipeline_tag values and tt-forge-models ModelTask
# snake_case strings so that either form dispatches correctly.
# ---------------------------------------------------------------------------
_TASK_FAMILY: dict[str, str] = {
    # HuggingFace pipeline_tag values
    "text-generation":              "causal_lm",
    "text2text-generation":         "seq2seq_lm",
    "fill-mask":                    "masked_lm",
    "question-answering":           "qa",
    "image-classification":         "image_cls",
    "object-detection":             "obj_det",
    "semantic-segmentation":        "segmentation",
    "image-segmentation":           "segmentation",
    "depth-estimation":             "depth",
    "automatic-speech-recognition": "asr",
    "audio-classification":         "audio_cls",
    "image-to-text":                "img2text",
    "visual-question-answering":    "img2text",
    "image-captioning":             "img2text",
    "text-to-speech":               "tts",
    "text-to-image":                "img_gen",
    # tt-forge-models ModelTask snake_case strings
    "causal_lm":                    "causal_lm",
    "causal_lm_with_past":          "causal_lm",
    "seq2seq_lm":                   "seq2seq_lm",
    "masked_lm":                    "masked_lm",
    "question_answering":           "qa",
    "image_classification":         "image_cls",
    "object_detection":             "obj_det",
    "semantic_segmentation":        "segmentation",
    "panoptic_segmentation":        "segmentation",
    "depth_estimation":             "depth",
    "automatic_speech_recognition": "asr",
    "audio_classification":         "audio_cls",
    "image_to_text":                "img2text",
    "visual_qa":                    "img2text",
    "image_captioning":             "img2text",
    "text_to_speech":               "tts",
    "image_generation":             "img_gen",
}


# ---------------------------------------------------------------------------
# Raw fallback — always succeeds
# ---------------------------------------------------------------------------

def _raw_fallback(output: Any) -> str:
    """Return shape/range info about *output*; never raises."""
    try:
        if hasattr(output, "shape"):
            shape = tuple(output.shape)
            try:
                import numpy as np
                arr = output.float().cpu().numpy()
                mn, mx = float(arr.min()), float(arr.max())
                return (
                    f"shape={shape} dtype={getattr(output, 'dtype', '?')} "
                    f"range=[{mn:.2f}, {mx:.2f}]"
                )
            except Exception:
                return f"shape={shape}"
        return f"output={type(output).__name__}"
    except Exception:
        return "decode failed — raw output"


# ---------------------------------------------------------------------------
# Per-family decoders
# ---------------------------------------------------------------------------

def _decode_causal_lm(output: Any, tokenizer: Any) -> str:
    """Decode autoregressive / seq2seq / masked LM output.

    Requires a tokenizer.  Without one we fall back to raw tensor info because
    token-id sequences are not human-readable on their own.
    """
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        # Expected shape: (batch, seq_len, vocab_size)
        if hasattr(output, "shape") and len(output.shape) == 3:
            # Greedy decode: pick argmax at each position for the first batch item.
            token_ids = output[0].argmax(dim=-1).tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return text[:100] if text else _raw_fallback(output)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_image_cls(output: Any) -> str:
    """Return top-3 class indices and confidence scores.

    Works with any output that exposes a .topk() method (real torch tensors
    and the MagicMock shims used in tests).
    """
    try:
        if hasattr(output, "topk"):
            scores, indices = output.topk(3)
            score_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            idx_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            # Unwrap a leading batch dimension if present (e.g. [[s0, s1, s2]]).
            if score_list and isinstance(score_list[0], list):
                score_list = score_list[0]
                idx_list = idx_list[0]
            parts = [
                f"class_{idx} {score:.2f}"
                for idx, score in zip(idx_list, score_list)
            ]
            return "top-3: " + ", ".join(parts)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_obj_det(output: Any) -> str:
    """Summarise object-detection output by shape or count."""
    try:
        if hasattr(output, "shape"):
            shape = tuple(output.shape)
            return f"detection output shape={shape}"
        if isinstance(output, (list, tuple)) and len(output) > 0:
            return f"{len(output)} detection(s) in output"
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_segmentation(output: Any) -> str:
    """Return segmentation map shape."""
    try:
        if hasattr(output, "shape"):
            return f"segmentation map shape={tuple(output.shape)}"
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_depth(output: Any) -> str:
    """Return depth map shape and value range (in metres when possible)."""
    try:
        if hasattr(output, "shape"):
            try:
                import numpy as np
                arr = output.float().cpu().numpy()
                return (
                    f"depth map shape={tuple(output.shape)} "
                    f"range=[{arr.min():.2f}m, {arr.max():.2f}m]"
                )
            except Exception:
                return f"depth map shape={tuple(output.shape)}"
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_asr(output: Any, tokenizer: Any) -> str:
    """Decode automatic speech recognition output token ids to text."""
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        # Expected shape: (batch, seq_len) — already decoded token ids.
        if hasattr(output, "shape") and len(output.shape) == 2:
            ids = output[0].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True)
            return text[:100] if text else _raw_fallback(output)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


# ---------------------------------------------------------------------------
# Dispatch table: family → (output, tokenizer, inputs) → str
# ---------------------------------------------------------------------------

_FAMILY_DECODERS: dict = {
    "causal_lm":    lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "seq2seq_lm":   lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "masked_lm":    lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "qa":           lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "image_cls":    lambda out, _tok, _inp: _decode_image_cls(out),
    "obj_det":      lambda out, _tok, _inp: _decode_obj_det(out),
    "segmentation": lambda out, _tok, _inp: _decode_segmentation(out),
    "depth":        lambda out, _tok, _inp: _decode_depth(out),
    "asr":          lambda out, tok, _inp: _decode_asr(out, tok),
    # Audio classification uses the same top-k presentation as image classification.
    "audio_cls":    lambda out, _tok, _inp: _decode_image_cls(out),
    "img2text":     lambda out, tok, _inp: _decode_causal_lm(out, tok),
    "tts":          lambda out, _tok, _inp: (
        f"audio output shape={tuple(out.shape) if hasattr(out, 'shape') else '?'}"
    ),
    "img_gen":      lambda out, _tok, _inp: (
        f"image output shape={tuple(out.shape) if hasattr(out, 'shape') else '?'}"
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode(
    output: Any,
    model_info: FrontierModelInfo,
    inputs: Any = None,
    tokenizer: Any = None,
) -> str:
    """Convert raw inference *output* to a human-readable artifact string.

    Parameters
    ----------
    output:
        Raw tensor (or tensor-like) returned by the compiled model.
    model_info:
        Metadata including the model's task string used for dispatch.
    inputs:
        Original model inputs (unused by most decoders; passed through for
        future decoders that need the prompt/image to contextualise output).
    tokenizer:
        Optional tokenizer used by language/speech decoders.

    Returns
    -------
    str
        Human-readable summary — never empty, never raises.
    """
    try:
        task_str = getattr(model_info, "task", "") or ""
        family = _TASK_FAMILY.get(task_str.lower(), None)
        if family and family in _FAMILY_DECODERS:
            return _FAMILY_DECODERS[family](output, tokenizer, inputs)
        # Unknown task family — fall through to raw fallback.
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)
