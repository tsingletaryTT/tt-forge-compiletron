# tests/test_author_dedup.py
"""Unit tests for author/family deduplication in build_queues."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from expedition import _family_key, _dedup_by_author_family


def _item(model_id, created="2024-01-01T00:00:00", params_b=0.0, downloads=1000):
    return {
        "model_id": model_id,
        "hf_created_at": created,
        "hf_params_b": params_b,
        "hf_downloads": downloads,
        "is_frontier": True,
    }


class TestFamilyKey:
    def test_strips_decimal_suffix(self):
        assert _family_key("icl-pruning-wanda-sparsity-0.5") == "icl-pruning-wanda-sparsity"

    def test_strips_integer_suffix(self):
        assert _family_key("model-name-2") == "model-name"

    def test_strips_version_prefix_v(self):
        assert _family_key("Mistral-7B-v0.1") == "mistral-7b"

    def test_no_numeric_suffix_unchanged(self):
        assert _family_key("bert-base-uncased") == "bert-base-uncased"

    def test_lowercase_applied(self):
        assert _family_key("BERT-Base") == "bert-base"

    def test_param_count_in_middle_not_stripped(self):
        # "7B" isn't at the end as a pure number token
        assert _family_key("llama-7b-instruct") == "llama-7b-instruct"

    def test_underscore_separator(self):
        assert _family_key("model_sparsity_0.9") == "model_sparsity"


class TestDedupByAuthorFamily:
    def test_single_model_passes_through(self):
        items = [_item("LiamCarter/icl-pruning-wanda-sparsity-0.5")]
        selected, dropped = _dedup_by_author_family(items)
        assert dropped == 0
        assert len(selected) == 1

    def test_sparsity_variants_collapsed(self):
        items = [
            _item("LiamCarter/icl-pruning-wanda-sparsity-0.5", created="2024-03-01"),
            _item("LiamCarter/icl-pruning-wanda-sparsity-0.1", created="2024-01-01"),
            _item("LiamCarter/icl-pruning-wanda-sparsity-0.9", created="2024-02-01"),
        ]
        selected, dropped = _dedup_by_author_family(items)
        assert dropped == 2
        assert len(selected) == 1
        # Most recent should win
        assert selected[0]["model_id"] == "LiamCarter/icl-pruning-wanda-sparsity-0.5"

    def test_different_authors_kept_separate(self):
        items = [
            _item("AuthorA/model-v1"),
            _item("AuthorB/model-v1"),
        ]
        selected, dropped = _dedup_by_author_family(items)
        assert dropped == 0
        assert len(selected) == 2

    def test_different_families_same_author_kept(self):
        items = [
            _item("google/bert-base-v1"),
            _item("google/vit-large-v1"),
        ]
        selected, dropped = _dedup_by_author_family(items)
        assert dropped == 0
        assert len(selected) == 2

    def test_prefers_size_near_sweet_spot(self):
        # With target 8B, sweet spot is 5.6B; 5B is closer than 1B
        items = [
            _item("org/model-v1", created="2024-01-01", params_b=1.0),
            _item("org/model-v2", created="2024-01-01", params_b=5.0),
        ]
        selected, dropped = _dedup_by_author_family(items, target_params_b=8.0)
        assert dropped == 1
        assert selected[0]["hf_params_b"] == 5.0

    def test_recency_beats_size(self):
        # Newer model wins even if slightly further from sweet spot
        items = [
            _item("org/model-v1", created="2024-01-01", params_b=5.0),
            _item("org/model-v2", created="2024-06-01", params_b=3.0),
        ]
        selected, dropped = _dedup_by_author_family(items, target_params_b=8.0)
        assert dropped == 1
        assert selected[0]["hf_created_at"] == "2024-06-01"

    def test_discovery_order_preserved(self):
        # Items are returned in original index order, not group order
        items = [
            _item("AuthorA/model-v1", created="2024-06-01"),
            _item("AuthorB/widget-v2", created="2024-05-01"),
            _item("AuthorA/model-v2", created="2024-04-01"),  # should be dropped
        ]
        selected, dropped = _dedup_by_author_family(items)
        assert dropped == 1
        # AuthorA's winner (model-v1) stays before AuthorB's widget
        ids = [x["model_id"] for x in selected]
        assert ids.index("AuthorA/model-v1") < ids.index("AuthorB/widget-v2")

    def test_no_drop_when_zero_items(self):
        selected, dropped = _dedup_by_author_family([])
        assert selected == []
        assert dropped == 0

    def test_version_suffix_with_v_prefix(self):
        items = [
            _item("mistralai/Mistral-7B-v0.1", created="2024-01-01"),
            _item("mistralai/Mistral-7B-v0.2", created="2024-06-01"),
            _item("mistralai/Mistral-7B-v0.3", created="2024-09-01"),
        ]
        selected, dropped = _dedup_by_author_family(items)
        assert dropped == 2
        assert selected[0]["model_id"] == "mistralai/Mistral-7B-v0.3"
