# tests/expedition/test_decoder.py
import pytest
from unittest.mock import MagicMock
from lib.expedition.decoder import decode, FrontierModelInfo


def _make_tensor(shape, values=None):
    """Return a MagicMock that behaves like a torch tensor for decode purposes."""
    t = MagicMock()
    t.shape = shape
    t.dtype = "torch.float32"
    t.__len__ = lambda self: shape[0]
    if values is not None:
        t.tolist.return_value = values
        t.argmax.return_value = MagicMock(item=lambda: 42)
        t.topk.return_value = (
            MagicMock(tolist=lambda: [0.9, 0.7, 0.5]),
            MagicMock(tolist=lambda: [42, 7, 99]),
        )
    import numpy as np
    t.float.return_value = t
    t.cpu.return_value = t
    t.numpy.return_value = np.zeros(shape)
    return t


class TestDecodeImageClassification:
    def test_returns_string(self):
        info = FrontierModelInfo(name="resnet50", task="image-classification")
        output = _make_tensor((1, 1000))
        output.topk.return_value = (
            MagicMock(tolist=lambda: [0.9, 0.7, 0.5]),
            MagicMock(tolist=lambda: [42, 7, 99]),
        )
        result = decode(output, info)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_shows_confidence(self):
        info = FrontierModelInfo(name="resnet50", task="image-classification")
        output = _make_tensor((1, 1000))
        output.topk.return_value = (
            MagicMock(tolist=lambda: [0.92, 0.71, 0.55]),
            MagicMock(tolist=lambda: [42, 7, 99]),
        )
        result = decode(output, info)
        assert "0.92" in result or "92" in result


class TestDecodeObjectDetection:
    def test_returns_string(self):
        info = FrontierModelInfo(name="yolov8", task="object-detection")
        # Simulate output dict-like with boxes and scores
        output = MagicMock()
        output.shape = (1, 100, 6)
        result = decode(output, info)
        assert isinstance(result, str)


class TestDecodeTextGeneration:
    def test_uses_tokenizer_when_available(self):
        import torch
        info = FrontierModelInfo(name="gpt2", task="text-generation")
        # Real tensor required — decoder calls torch.topk on last-position logits.
        output = torch.zeros(1, 10, 50257)
        output[0, -1, 7] = 10.0   # make token 7 the clear top prediction
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "silicon"
        result = decode(output, info, tokenizer=tokenizer)
        assert "→" in result
        assert "silicon" in result

    def test_falls_back_without_tokenizer(self):
        import torch
        info = FrontierModelInfo(name="gpt2", task="text-generation")
        output = torch.zeros(1, 10, 50257)
        result = decode(output, info, tokenizer=None)
        assert isinstance(result, str)


class TestDecodeRawFallback:
    def test_unknown_task_returns_shape_info(self):
        info = FrontierModelInfo(name="mystery", task="unknown-task-xyz")
        output = _make_tensor((1, 256, 256))
        result = decode(output, info)
        assert "shape" in result.lower() or "256" in result

    def test_exception_in_decode_returns_fallback(self):
        info = FrontierModelInfo(name="crash", task="image-classification")
        output = MagicMock(spec=[])  # no attributes — will raise on access
        result = decode(output, info)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFrontierModelInfo:
    def test_has_task_attribute(self):
        info = FrontierModelInfo(name="test", task="text-generation")
        assert info.task == "text-generation"
        assert info.source == "huggingface"


class TestDecodeSegmentation:
    def test_returns_string(self):
        info = FrontierModelInfo(name="seg-model", task="semantic-segmentation")
        output = _make_tensor((1, 10, 256, 256))
        result = decode(output, info)
        assert isinstance(result, str)
        assert len(result) > 0


class TestDecodeASR:
    def test_with_tokenizer(self):
        info = FrontierModelInfo(name="whisper", task="automatic-speech-recognition")
        output = _make_tensor((1, 80))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "hello world"
        result = decode(output, info, tokenizer=tokenizer)
        assert isinstance(result, str)

    def test_without_tokenizer_falls_back(self):
        info = FrontierModelInfo(name="whisper", task="automatic-speech-recognition")
        output = _make_tensor((1, 80))
        result = decode(output, info, tokenizer=None)
        assert isinstance(result, str)


class TestDecodeDepth:
    def test_returns_string(self):
        info = FrontierModelInfo(name="depth-model", task="depth-estimation")
        output = _make_tensor((1, 1, 256, 256))
        result = decode(output, info)
        assert isinstance(result, str)


class TestDecodeImageGen:
    def test_returns_range_info(self):
        info = FrontierModelInfo(name="stable-diffusion", task="text-to-image")
        output = _make_tensor((1, 3, 512, 512))
        result = decode(output, info)
        assert isinstance(result, str)
        # image_gen now uses _raw_fallback which includes range
        assert "shape" in result.lower()


class TestTaskAliases:
    def test_fill_mask_dispatches(self):
        info = FrontierModelInfo(name="bert", task="fill-mask")
        output = _make_tensor((1, 10, 30000))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "cat"
        tokenizer.mask_token_id = 103
        result = decode(output, info, tokenizer=tokenizer)
        assert isinstance(result, str)

    def test_qa_dispatch(self):
        info = FrontierModelInfo(name="bert-qa", task="question-answering")
        # QA output: tuple of (start_logits, end_logits) each shape (1, seq_len)
        start_logits = MagicMock()
        start_logits.__getitem__ = lambda self, i: MagicMock(argmax=lambda: MagicMock(item=lambda: 5))
        end_logits = MagicMock()
        end_logits.__getitem__ = lambda self, i: MagicMock(argmax=lambda: MagicMock(item=lambda: 8))
        result = decode((start_logits, end_logits), info)
        assert isinstance(result, str)

    def test_visual_qa_dispatches(self):
        info = FrontierModelInfo(name="blip", task="visual-question-answering")
        output = _make_tensor((1, 20, 30000))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "a cat sitting"
        result = decode(output, info, tokenizer=tokenizer)
        assert isinstance(result, str)
