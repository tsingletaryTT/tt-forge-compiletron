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


from lib.expedition.expedition_worker import _run_bench_passes
import time


class TestRunBenchPasses:
    def _make_compiled(self, sleep_s=0.01):
        """Return a callable that fakes a compiled module with known latency."""
        import torch

        def fake_compiled(*args):
            time.sleep(sleep_s)
            return torch.zeros(1, 32, 50257)

        return fake_compiled

    def test_returns_expected_keys(self):
        compiled = self._make_compiled(sleep_s=0.01)
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=3, task="text-generation")
        assert set(result.keys()) == {
            "bench_passes", "infer_p50_s", "infer_p95_s",
            "throughput_p50", "throughput_p95",
        }

    def test_bench_passes_count_matches(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=5, task="text-generation")
        assert result["bench_passes"] == 5

    def test_p50_le_p95(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=5, task="text-generation")
        assert result["infer_p50_s"] <= result["infer_p95_s"]

    def test_throughput_unit_tokens_per_sec(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=3, task="text-generation")
        assert result["throughput_p50"] > 0

    def test_zero_passes_returns_empty(self):
        compiled = self._make_compiled()
        inputs = [torch.randint(0, 1000, (1, 32))]
        result = _run_bench_passes(compiled, inputs, n_passes=0, task="text-generation")
        assert result == {}

    def test_crashing_compiled_returns_empty(self):
        def crashing(*args):
            raise RuntimeError("simulated crash")

        import torch
        inputs = [torch.zeros(1, 32)]
        result = _run_bench_passes(crashing, inputs, n_passes=5, task="text-generation")
        assert result == {}
