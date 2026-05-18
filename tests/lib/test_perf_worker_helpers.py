import sys
import types
import torch
import pytest

# Stub out forge so the worker module can be imported without forge installed.
_forge_stub = types.ModuleType("forge")
sys.modules.setdefault("forge", _forge_stub)

from lib.expedition.expedition_worker import _compute_throughput


class TestComputeThroughput:
    def _tensor(self, shape):
        return torch.zeros(*shape)

    def test_text_generation_tokens_per_sec(self):
        output = self._tensor((1, 64, 50257))  # batch=1, seq=64, vocab
        tput, unit = _compute_throughput("text-generation", output, infer_s=2.0)
        assert unit == "tokens/sec"
        assert tput == pytest.approx(32.0)  # 64 tokens / 2s

    def test_nlp_causal_lm_tokens_per_sec(self):
        output = self._tensor((1, 32, 32000))
        tput, unit = _compute_throughput("nlp_causal_lm", output, infer_s=1.0)
        assert unit == "tokens/sec"
        assert tput == pytest.approx(32.0)

    def test_fill_mask_tokens_per_sec(self):
        output = self._tensor((1, 32, 30000))
        tput, unit = _compute_throughput("fill-mask", output, infer_s=0.5)
        assert unit == "tokens/sec"
        assert tput == pytest.approx(64.0)

    def test_image_classification_ms_per_sample(self):
        output = self._tensor((1, 1000))
        tput, unit = _compute_throughput("image-classification", output, infer_s=0.1)
        assert unit == "ms/sample"
        assert tput == pytest.approx(100.0)

    def test_cv_image_cls_ms_per_sample(self):
        output = self._tensor((1, 1000))
        tput, unit = _compute_throughput("cv_image_cls", output, infer_s=0.05)
        assert unit == "ms/sample"
        assert tput == pytest.approx(50.0)

    def test_zero_infer_s_returns_empty(self):
        output = self._tensor((1, 32, 50257))
        tput, unit = _compute_throughput("text-generation", output, infer_s=0.0)
        assert tput == 0.0
        assert unit == ""

    def test_none_output_returns_empty(self):
        tput, unit = _compute_throughput("text-generation", None, infer_s=1.0)
        assert tput == 0.0
        assert unit == ""

    def test_nlp_embed_gen_uses_ms_per_sample(self):
        output = self._tensor((1, 32, 1024))
        tput, unit = _compute_throughput("nlp_embed_gen", output, infer_s=0.2)
        assert unit == "ms/sample"

    def test_nlp_qa_uses_ms_per_sample(self):
        output = self._tensor((1, 32))
        tput, unit = _compute_throughput("nlp_qa", output, infer_s=0.1)
        assert unit == "ms/sample"
