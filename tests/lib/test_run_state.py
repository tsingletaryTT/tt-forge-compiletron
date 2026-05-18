from lib.expedition.run_state import ModelResult


class TestModelResultInferTime:
    def _make_row(self, infer_time="0.42"):
        return {
            "model": "gpt2/pytorch",
            "status": "success",
            "pts": "250",
            "compile_time": "10.5",
            "infer_time": infer_time,
            "artifact": "shape=(1,32,50257)",
            "first_ever": "True",
            "first_voice": "False",
            "error": "",
        }

    def test_from_csv_row_parses_infer_time(self):
        row = self._make_row(infer_time="0.42")
        result = ModelResult.from_csv_row(row, chip_id=0, rarity="common", streak=1)
        assert result.infer_time == 0.42

    def test_from_csv_row_defaults_infer_time_to_zero(self):
        row = self._make_row()
        del row["infer_time"]
        result = ModelResult.from_csv_row(row, chip_id=0, rarity="common", streak=1)
        assert result.infer_time == 0.0

    def test_from_csv_row_handles_empty_infer_time(self):
        row = self._make_row(infer_time="")
        result = ModelResult.from_csv_row(row, chip_id=0, rarity="common", streak=1)
        assert result.infer_time == 0.0

    def test_infer_time_field_exists_on_model_result(self):
        result = ModelResult(
            chip_id=0, model_id="x", status="success", pts=100,
            compile_time=5.0, infer_time=0.5, artifact="", first_ever=False,
            first_voice=False, error="", rarity="common", streak=1,
        )
        assert result.infer_time == 0.5
