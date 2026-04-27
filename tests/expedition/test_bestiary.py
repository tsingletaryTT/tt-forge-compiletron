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
        assert entry["attempts"] == 1

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
        assert entry["attempts"] == 3

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
