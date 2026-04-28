# lib/expedition/sampler.py
"""Themed sample input selector for First Voice inference.

Returns a ready-to-use dict describing one sample input for a given
HuggingFace pipeline task.  All returned paths are absolute so the
caller never needs to know where the repo lives.

Returned dict shape:
  {
    "input_type": "text" | "image" | "audio",
    "data":        str (prompt/path),
    "description": str  # human-readable label for journal entries
  }

For question-answering tasks, "data" is a dict {"context": str, "question": str}.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path


# Resolve asset directories relative to this file (lib/expedition/sampler.py → repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SAMPLES_DIR  = _REPO_ROOT / "data" / "samples"
_PROMPTS_FILE = _SAMPLES_DIR / "text" / "prompts.json"
_IMAGES_DIR   = _SAMPLES_DIR / "images"
_AUDIO_DIR    = _SAMPLES_DIR / "audio"


def _load_prompts() -> dict:
    """Load the themed prompt bank.  Returns empty dict on any failure."""
    try:
        with open(_PROMPTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Strip the human-readable comment key before callers see the dict.
        data.pop("_comment", None)
        return data
    except Exception:
        return {}


def _image_paths() -> list[str]:
    """Return a list of absolute paths to sample images (JPEG files only)."""
    if not _IMAGES_DIR.exists():
        return []
    return [str(p) for p in sorted(_IMAGES_DIR.glob("*.jpg"))]


def _audio_paths() -> list[str]:
    """Return a list of absolute paths to sample audio files."""
    if not _AUDIO_DIR.exists():
        return []
    exts = {".mp3", ".wav", ".flac", ".ogg"}
    return [str(p) for p in sorted(_AUDIO_DIR.iterdir()) if p.suffix.lower() in exts]


# Maps HF pipeline_tag → which sample pool to draw from.
# "text" tasks draw a prompt string; "image" tasks draw an image path;
# "audio" tasks draw an audio file path.
_TASK_INPUT_TYPE: dict[str, str] = {
    "text-generation":              "text",
    "text2text-generation":         "text",
    "fill-mask":                    "text",
    "question-answering":           "text",
    "image-classification":         "image",
    "object-detection":             "image",
    "semantic-segmentation":        "image",
    "image-segmentation":           "image",
    "depth-estimation":             "image",
    "image-to-text":                "image",
    "visual-question-answering":    "image",
    "image-captioning":             "image",
    "automatic-speech-recognition": "audio",
    "audio-classification":         "audio",
}


def get_sample(task: str) -> dict | None:
    """Return a themed sample dict for the given pipeline task.

    Returns None when:
    - The task is not in the known task-to-input-type map.
    - No sample assets exist for that input type (empty directories or missing files).

    The "data" field holds:
    - str (prompt or file path) for all types except question-answering.
    - dict {"context": str, "question": str} for question-answering.
    """
    input_type = _TASK_INPUT_TYPE.get(task)
    if input_type is None:
        return None

    if input_type == "text":
        prompts = _load_prompts()
        pool = prompts.get(task, [])
        # Fall back to text-generation prompts for unknown text tasks.
        if not pool:
            pool = prompts.get("text-generation", [])
        if not pool:
            return None
        chosen = random.choice(pool)
        if isinstance(chosen, dict):
            # question-answering entries are {"context": …, "question": …}
            description = f"Q: {chosen.get('question', '')[:60]}"
        else:
            description = str(chosen)[:60]
        return {"input_type": "text", "data": chosen, "description": description}

    if input_type == "image":
        paths = _image_paths()
        if not paths:
            return None
        chosen = random.choice(paths)
        description = os.path.basename(chosen).replace("_", " ").rsplit(".", 1)[0]
        return {"input_type": "image", "data": chosen, "description": description}

    if input_type == "audio":
        paths = _audio_paths()
        if not paths:
            return None
        # Prefer WAV files (better compatibility with most ASR processors); fall back to MP3.
        wav_paths = [p for p in paths if p.endswith(".wav")]
        chosen = random.choice(wav_paths if wav_paths else paths)
        description = os.path.basename(chosen).replace("_", " ").rsplit(".", 1)[0]
        return {"input_type": "audio", "data": chosen, "description": description}

    return None


def make_tensor_input(sample: dict, seq_len: int = 32, tokenizer=None):
    """Convert a sample dict into a torch tensor for forge inference.

    For text samples: tokenizes with the provided tokenizer, pads/truncates to
    seq_len.  ``tokenizer`` must be the model's own tokenizer — using a
    mismatched tokenizer produces input IDs in the wrong vocabulary, which the
    compiled model decodes as gibberish.  If no tokenizer is provided and the
    input_type is "text", returns (None, "no tokenizer") so the caller can skip
    First Voice rather than produce garbage output.

    For image samples: opens the image with PIL and applies standard ImageNet
                       normalisation (224×224, RGB).
    For audio samples: loads the audio with torchaudio, resamples to 16 kHz mono.

    Returns (tensor, description_suffix) where description_suffix is a short
    string explaining what was done — useful for journal entries.
    Returns (None, reason) when the conversion cannot proceed.
    """
    if sample["input_type"] == "text":
        return _text_tensor(sample["data"], seq_len, tokenizer)
    if sample["input_type"] == "image":
        return _image_tensor(sample["data"])
    if sample["input_type"] == "audio":
        return _audio_tensor(sample["data"])
    return None, "unknown input type"


def _text_tensor(data, seq_len: int, tokenizer):
    """Tokenize text data into a fixed-length integer tensor.

    Requires the model's own tokenizer — the calling code in
    _attempt_first_voice loads it by model_id before calling here.
    Returns (None, reason) rather than falling back to a wrong-vocab tokenizer,
    because mismatched input IDs produce meaningless decoded output.
    """
    if tokenizer is None:
        return None, "no tokenizer — skipping text First Voice"
    try:
        if isinstance(data, dict):
            # question-answering: encode context + question together
            text = data.get("context", "") + " " + data.get("question", "")
        else:
            text = str(data)
        # Causal LM tokenizers (LLaMA, Mistral, Qwen, Falcon, GPT-2, …) have no
        # pad_token because autoregressive training never needs padding.  Set it
        # to eos_token so padding="max_length" works without raising an error.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        enc = tokenizer(text, return_tensors="pt", max_length=seq_len,
                        padding="max_length", truncation=True)
        return enc["input_ids"], f"tokenized ({seq_len} tokens)"
    except Exception as e:
        return None, f"tokenizer error: {e}"


def _image_tensor(path: str):
    """Load and normalise an image for ImageNet-style vision models."""
    import torch
    try:
        from PIL import Image
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0), os.path.basename(path)
    except Exception:
        return torch.randn(1, 3, 224, 224), "random fallback"


def _audio_tensor(path: str):
    """Load an audio file and return a 16kHz mono float tensor."""
    import torch
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        return waveform, os.path.basename(path)
    except Exception:
        return torch.randn(1, 16000), "random fallback"
