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
        info = FrontierModelInfo(name="gpt2", task="text-generation")
        output = _make_tensor((1, 50, 50257))
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "Hello world from GPT-2"
        result = decode(output, info, tokenizer=tokenizer)
        assert "Hello world" in result

    def test_falls_back_without_tokenizer(self):
        info = FrontierModelInfo(name="gpt2", task="text-generation")
        output = _make_tensor((1, 50, 50257))
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
