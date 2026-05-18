import json
import pytest
from pathlib import Path
from lib.expedition.bestiary import Bestiary

@pytest.fixture
def tmp_bestiary(tmp_path):
    return Bestiary(path=tmp_path / "bestiary.json")

class TestBestiaryInit:
    def test_creates_empty_on_missing_file(self, tmp_bestiary):
        assert tmp_bestiary.compiled == {}
        assert tmp_bestiary.failed == {}
        assert tmp_bestiary.chip_totals == {}

    def test_loads_existing_file(self, tmp_path):
        data = {"compiled": {"bert": {"artifact": "hello"}}, "failed": {}, "chip_totals": {}}
        (tmp_path / "bestiary.json").write_text(json.dumps(data))
        b = Bestiary(path=tmp_path / "bestiary.json")
        assert "bert" in b.compiled

class TestBestiaryRecordSuccess:
    def test_adds_to_compiled(self, tmp_bestiary):
        tmp_bestiary.record_success(
            model_id="openai/whisper-large-v3",
            chip=0, run=1, time_s=31.2,
            task="automatic_speech_recognition",
            source="huggingface",
            rarity="common",
            hf_downloads=5_000_000,
            hf_created_at="2023-09-01T00:00:00",
            artifact="Mr. Gorbachev, tear down this wall.",
        )
        assert "openai/whisper-large-v3" in tmp_bestiary.compiled
        entry = tmp_bestiary.compiled["openai/whisper-large-v3"]
        assert entry["first_chip"] == 0
        assert entry["run"] == 1
        assert entry["artifact"] == "Mr. Gorbachev, tear down this wall."
        assert entry["successes"] == 1
        assert "attempts" not in entry  # only successes tracked; failures tracked in failed dict

    def test_repeat_success_increments_counters(self, tmp_bestiary):
        for _ in range(3):
            tmp_bestiary.record_success(
                model_id="openai/whisper-large-v3",
                chip=0, run=1, time_s=30.0,
                task="automatic_speech_recognition",
                source="huggingface",
                rarity="common",
                hf_downloads=5_000_000,
                hf_created_at="2023-09-01T00:00:00",
                artifact="test",
            )
        entry = tmp_bestiary.compiled["openai/whisper-large-v3"]
        assert entry["successes"] == 3

    def test_best_time_updated(self, tmp_bestiary):
        tmp_bestiary.record_success(
            model_id="m", chip=0, run=1, time_s=50.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        tmp_bestiary.record_success(
            model_id="m", chip=1, run=2, time_s=30.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="b",
        )
        assert tmp_bestiary.compiled["m"]["best_time_s"] == 30.0

class TestBestiaryRecordFailure:
    def test_adds_to_failed(self, tmp_bestiary):
        tmp_bestiary.record_failure(
            model_id="mistralai/Mistral-7B-v0.3",
            run=5, error="RuntimeError: rotary embedding shape mismatch",
        )
        assert "mistralai/Mistral-7B-v0.3" in tmp_bestiary.failed
        entry = tmp_bestiary.failed["mistralai/Mistral-7B-v0.3"]
        assert entry["attempts"] == 1
        assert "rotary" in entry["last_error"]

    def test_repeat_failure_increments_attempts(self, tmp_bestiary):
        for _ in range(3):
            tmp_bestiary.record_failure(
                model_id="bad-model", run=1, error="timeout",
            )
        assert tmp_bestiary.failed["bad-model"]["attempts"] == 3

class TestBestiaryChipTotals:
    def test_add_chip_points(self, tmp_bestiary):
        tmp_bestiary.add_chip_points(chip=0, pts=150, first_ever=True, streak=5)
        assert tmp_bestiary.chip_totals["0"]["pts"] == 150
        assert tmp_bestiary.chip_totals["0"]["first_evers"] == 1
        assert tmp_bestiary.chip_totals["0"]["best_streak"] == 5

    def test_accumulates(self, tmp_bestiary):
        tmp_bestiary.add_chip_points(chip=0, pts=100, first_ever=True, streak=3)
        tmp_bestiary.add_chip_points(chip=0, pts=50, first_ever=False, streak=4)
        assert tmp_bestiary.chip_totals["0"]["pts"] == 150
        assert tmp_bestiary.chip_totals["0"]["first_evers"] == 1
        assert tmp_bestiary.chip_totals["0"]["best_streak"] == 4

class TestBestiaryIsCompiled:
    def test_not_compiled(self, tmp_bestiary):
        assert tmp_bestiary.is_compiled("unknown/model") is False

    def test_compiled_after_success(self, tmp_bestiary):
        tmp_bestiary.record_success(
            model_id="x", chip=0, run=1, time_s=1.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="out",
        )
        assert tmp_bestiary.is_compiled("x") is True

    def test_success_after_failure_dual_membership(self, tmp_bestiary):
        # compiled and failed are not mutually exclusive; document this behavior
        tmp_bestiary.record_failure("m", run=1, error="oom")
        tmp_bestiary.record_success(
            model_id="m", chip=0, run=2, time_s=30.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        assert tmp_bestiary.is_compiled("m")
        assert "m" in tmp_bestiary.failed

class TestBestiaryPersistence:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "bestiary.json"
        b1 = Bestiary(path=path)
        b1.record_success(
            model_id="m", chip=0, run=1, time_s=1.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        b1.save()
        b2 = Bestiary(path=path)
        assert "m" in b2.compiled

    def test_file_is_valid_json(self, tmp_path):
        path = tmp_path / "bestiary.json"
        b = Bestiary(path=path)
        b.save()
        data = json.loads(path.read_text())
        assert "compiled" in data
        assert "failed" in data
        assert "chip_totals" in data


class TestBestiaryPerfFields:
    def _make_success(self, b, model_id="m", compile_s=5.0, infer_s=0.5,
                      throughput=12.0, throughput_unit="tokens/sec"):
        b.record_success(
            model_id=model_id, chip=0, run=1, time_s=compile_s + infer_s,
            task="text-generation", source="hf", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="x",
            compile_s=compile_s, infer_s=infer_s,
            throughput=throughput, throughput_unit=throughput_unit,
        )

    def test_first_success_stores_perf_fields(self, tmp_bestiary):
        self._make_success(tmp_bestiary)
        e = tmp_bestiary.compiled["m"]
        assert e["best_compile_s"] == 5.0
        assert e["best_infer_s"] == 0.5
        assert e["best_throughput"] == 12.0
        assert e["throughput_unit"] == "tokens/sec"

    def test_tokens_per_sec_higher_is_better(self, tmp_bestiary):
        self._make_success(tmp_bestiary, throughput=10.0, throughput_unit="tokens/sec")
        self._make_success(tmp_bestiary, throughput=15.0, throughput_unit="tokens/sec")
        assert tmp_bestiary.compiled["m"]["best_throughput"] == 15.0

    def test_ms_per_sample_lower_is_better(self, tmp_bestiary):
        self._make_success(tmp_bestiary, throughput=100.0, throughput_unit="ms/sample")
        self._make_success(tmp_bestiary, throughput=80.0, throughput_unit="ms/sample")
        assert tmp_bestiary.compiled["m"]["best_throughput"] == 80.0

    def test_best_compile_s_tracks_minimum(self, tmp_bestiary):
        self._make_success(tmp_bestiary, compile_s=8.0)
        self._make_success(tmp_bestiary, compile_s=5.0)
        self._make_success(tmp_bestiary, compile_s=6.0)
        assert tmp_bestiary.compiled["m"]["best_compile_s"] == 5.0

    def test_zero_values_not_stored(self, tmp_bestiary):
        b = tmp_bestiary
        b.record_success(
            model_id="x", chip=0, run=1, time_s=10.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        assert "best_compile_s" not in b.compiled["x"]
        assert "best_infer_s" not in b.compiled["x"]
        assert "best_throughput" not in b.compiled["x"]

    def test_existing_tests_still_pass_without_new_args(self, tmp_bestiary):
        tmp_bestiary.record_success(
            model_id="legacy", chip=0, run=1, time_s=30.0,
            task="t", source="s", rarity="common",
            hf_downloads=None, hf_created_at=None, artifact="a",
        )
        assert "legacy" in tmp_bestiary.compiled


class TestAppendPerfRecord:
    def test_appends_jsonl_line(self, tmp_bestiary):
        record = {"model_id": "gpt2/pytorch", "run": 1, "compile_s": 10.2,
                  "infer_s": 2.3, "throughput": 13.9, "throughput_unit": "tokens/sec"}
        tmp_bestiary.append_perf_record(record)
        perf_path = tmp_bestiary.path.parent / "perf_history.jsonl"
        assert perf_path.exists()
        lines = perf_path.read_text().strip().split("\n")
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["model_id"] == "gpt2/pytorch"
        assert loaded["compile_s"] == 10.2

    def test_appends_multiple_records(self, tmp_bestiary):
        for i in range(3):
            tmp_bestiary.append_perf_record({"run": i})
        perf_path = tmp_bestiary.path.parent / "perf_history.jsonl"
        lines = perf_path.read_text().strip().split("\n")
        assert len(lines) == 3


class TestBestiaryArtifacts:
    def test_save_artifact_creates_file(self, tmp_path, tmp_bestiary):
        out = tmp_bestiary.save_artifact(
            model_id="openai/whisper-large-v3",
            task="automatic_speech_recognition",
            compiled_at="2026-04-27T14:00:00",
            chip=1,
            run=3,
            artifact_text="Mr. Gorbachev, tear down this wall.",
            artifacts_dir=tmp_path / "artifacts",
        )
        assert out.exists()
        content = out.read_text()
        assert "openai/whisper-large-v3" in content
        assert "chip-1" in content
        assert "run-3" in content
        assert "Mr. Gorbachev" in content

    def test_save_artifact_header_format(self, tmp_path, tmp_bestiary):
        out = tmp_bestiary.save_artifact(
            model_id="myorg/mymodel",
            task="text-generation",
            compiled_at="2026-04-27T12:00:00",
            chip=0,
            run=1,
            artifact_text="hello world",
            artifacts_dir=tmp_path / "artifacts",
        )
        lines = out.read_text().splitlines()
        assert lines[0].startswith("myorg/mymodel")
        assert "text-generation" in lines[0]
        assert "chip-0" in lines[0]
        assert "run-1" in lines[0]
        assert lines[1] == "hello world"

    def test_load_artifact_returns_content(self, tmp_path, tmp_bestiary):
        tmp_bestiary.save_artifact(
            model_id="openai/whisper-large-v3",
            task="asr",
            compiled_at="now",
            chip=0,
            run=1,
            artifact_text="tear down this wall",
            artifacts_dir=tmp_path / "artifacts",
        )
        content = tmp_bestiary.load_artifact("openai/whisper-large-v3", artifacts_dir=tmp_path / "artifacts")
        assert content is not None
        assert "tear down this wall" in content

    def test_load_artifact_missing_returns_none(self, tmp_path, tmp_bestiary):
        result = tmp_bestiary.load_artifact("no/such/model", artifacts_dir=tmp_path / "artifacts")
        assert result is None

    def test_sanitize_model_id_slash(self, tmp_path, tmp_bestiary):
        out = tmp_bestiary.save_artifact(
            model_id="openai/whisper-large-v3",
            task="asr", compiled_at="now", chip=0, run=1,
            artifact_text="x", artifacts_dir=tmp_path / "artifacts",
        )
        assert "openai_whisper-large-v3.txt" == out.name

    def test_failure_artifact_with_error_string(self, tmp_path, tmp_bestiary):
        # Callers (expedition_worker) use save_artifact for epic fails too
        error = "RuntimeError: rotary embedding shape mismatch"
        out = tmp_bestiary.save_artifact(
            model_id="mistralai/Mistral-7B-v0.3",
            task="text-generation",
            compiled_at="2026-04-27T15:00:00",
            chip=3,
            run=5,
            artifact_text=error,
            artifacts_dir=tmp_path / "artifacts",
        )
        assert out.exists()
        content = out.read_text()
        assert "rotary embedding" in content
        assert "chip-3" in content
