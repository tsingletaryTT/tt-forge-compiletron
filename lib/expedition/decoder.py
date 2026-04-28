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

    Uses last-position logits to show the model's top-3 predicted next tokens
    (with probabilities) given the prompt.  Full-sequence argmax produces
    garbled "autoregressive echo" output because each position independently
    predicts the next token from its own prefix — the predictions are not chained.
    """
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        import torch
        # Expected shape: (batch, seq_len, vocab_size)
        if hasattr(output, "shape") and len(output.shape) == 3:
            # Last-position logits: what the model predicts comes after the full prompt.
            last_logits = output[0, -1, :]
            topk = torch.topk(last_logits, k=min(5, last_logits.shape[-1]))
            probs = torch.softmax(last_logits, dim=-1)[topk.indices]
            words = []
            for t, p in zip(topk.indices.tolist(), probs.tolist()):
                w = tokenizer.decode([t], skip_special_tokens=True).strip()
                if w:
                    words.append(f"{w} ({p:.0%})")
            if words:
                return "→ " + " | ".join(words[:3])
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
    """Return unique class labels present in the segmentation mask.

    For a (batch, classes, H, W) logit tensor we argmax over the class
    dimension to get the predicted class per pixel, then report the sorted
    set of unique class ids.  Falls back to shape-only info if numpy is
    unavailable or the tensor has an unexpected layout.
    """
    try:
        if hasattr(output, "shape"):
            try:
                import numpy as np
                arr = output.float().cpu().numpy()
                # Argmax over class dim: (batch, classes, H, W) → (batch, H, W)
                if arr.ndim == 4:
                    mask = arr.argmax(axis=1)
                elif arr.ndim == 3:
                    # (classes, H, W) → (H, W)
                    mask = arr.argmax(axis=0)
                else:
                    mask = arr
                unique_classes = sorted(np.unique(mask).astype(int).tolist())
                return f"classes: {unique_classes} shape={tuple(output.shape)}"
            except Exception:
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


def _decode_masked_lm(output: Any, tokenizer: Any, inputs: Any) -> str:
    """Decode the top predicted token at the [MASK] position.

    Looks up the [MASK] token id in *inputs* so we can report the single most
    likely replacement token rather than greedy-decoding the full sequence.
    Falls back to full greedy decode if the mask position cannot be found, and
    to raw tensor info if no tokenizer is supplied.
    """
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        # Expected shape: (batch, seq_len, vocab_size)
        if hasattr(output, "shape") and len(output.shape) == 3:
            # Try to find the [MASK] position from the input ids.
            mask_pos = None
            if inputs is not None:
                input_ids = None
                if isinstance(inputs, dict) and "input_ids" in inputs:
                    input_ids = inputs["input_ids"]
                elif hasattr(inputs, "input_ids"):
                    input_ids = inputs.input_ids
                if input_ids is not None and hasattr(tokenizer, "mask_token_id"):
                    mask_id = tokenizer.mask_token_id
                    # Handle (batch, seq) tensors, (seq,) 1-D tensors, and plain lists.
                    ndim = len(input_ids.shape) if hasattr(input_ids, "shape") else 1
                    seq = input_ids[0] if ndim > 1 else input_ids
                    ids = seq.tolist() if hasattr(seq, "tolist") else list(seq)
                    if mask_id in ids:
                        mask_pos = ids.index(mask_id)

            if mask_pos is not None:
                # Decode only the single position that was masked.
                logits_at_mask = output[0, mask_pos]
                top_token_id = logits_at_mask.argmax().item()
                token = tokenizer.decode([top_token_id], skip_special_tokens=True)
                return f"predicted [MASK]: {token!r}"

            # Mask position unknown — fall back to greedy full-sequence decode.
            token_ids = output[0].argmax(dim=-1).tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            return text[:100] if text else _raw_fallback(output)
        return _raw_fallback(output)
    except Exception:
        return _raw_fallback(output)


def _decode_qa(output: Any, tokenizer: Any, inputs: Any) -> str:
    """Extract the answer span from a QA model's start/end position logits.

    QA models (e.g. BERT for QA) return two (batch, seq_len) tensors —
    *start_logits* and *end_logits*.  We take the argmax of each to identify
    the answer token span, then decode those tokens if input_ids are available.
    """
    if tokenizer is None:
        return _raw_fallback(output)
    try:
        # Support both tuple/list output and attribute-based output objects.
        if isinstance(output, (tuple, list)) and len(output) == 2:
            start_logits, end_logits = output
        elif hasattr(output, "start_logits") and hasattr(output, "end_logits"):
            start_logits = output.start_logits
            end_logits = output.end_logits
        else:
            # Unknown format — fall back to causal LM decode as best effort.
            return _decode_causal_lm(output, tokenizer)

        start_idx = start_logits[0].argmax().item()
        end_idx = end_logits[0].argmax().item()
        # Ensure valid span (end must not precede start).
        if end_idx < start_idx:
            end_idx = start_idx

        # Decode the answer span from input_ids if available.
        if inputs is not None:
            input_ids = None
            if isinstance(inputs, dict) and "input_ids" in inputs:
                input_ids = inputs["input_ids"]
            elif hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
            if input_ids is not None:
                ids = (
                    input_ids[0].tolist()
                    if hasattr(input_ids[0], "tolist")
                    else list(input_ids[0])
                )
                # Clamp to sequence length to guard against out-of-range predictions.
                seq_len = len(ids)
                start_idx = min(start_idx, seq_len - 1)
                end_idx = min(end_idx, seq_len - 1)
                answer_ids = ids[start_idx : end_idx + 1]
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
                if answer:
                    return f"answer: {answer!r} (tokens {start_idx}–{end_idx})"
                return f"answer span empty (tokens {start_idx}–{end_idx}, seq_len={seq_len})"

        return f"answer span: tokens {start_idx}–{end_idx}"
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
    "masked_lm":    lambda out, tok, inp: _decode_masked_lm(out, tok, inp),
    "qa":           lambda out, tok, inp: _decode_qa(out, tok, inp),
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
    # img_gen uses _raw_fallback so it reports pixel value range as required.
    "img_gen":      lambda out, _tok, _inp: _raw_fallback(out),
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
        Original model inputs. Used by masked_lm (to locate the [MASK] token
        position) and qa (to decode the answer span text). Unused by image
        and audio decoders. Pass None if inputs are unavailable — those two
        decoders will fall back gracefully.
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
