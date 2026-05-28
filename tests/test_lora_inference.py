"""Tests for LoRA NER inference and dataset utilities.

Uses a tiny BERT config so tests run quickly without downloading large models.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.finetune.dataset import (
    LABEL2ID,
    LABEL_LIST,
    NERExample,
    NERLabel,
    FinancialNERDataset,
    generate_synthetic_examples,
    save_examples_to_jsonl,
)
from src.finetune.inference import Entity, _aggregate_entities, _deduplicate


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestNERLabel:
    def test_all_labels_present(self):
        expected = {"O", "B-ORG", "I-ORG", "B-MONEY", "I-MONEY", "B-DATE", "I-DATE", "B-PERCENT", "I-PERCENT"}
        assert set(LABEL_LIST) == expected

    def test_label2id_mapping(self):
        assert LABEL2ID["O"] == 0
        assert "B-ORG" in LABEL2ID
        assert "B-MONEY" in LABEL2ID

    def test_round_trip(self):
        from src.finetune.dataset import ID2LABEL
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label


class TestNERExample:
    def test_valid_example(self):
        tokens = ["Goldman", "Sachs", "earned", "$", "10B"]
        tags = [LABEL2ID["B-ORG"], LABEL2ID["I-ORG"], LABEL2ID["O"], LABEL2ID["B-MONEY"], LABEL2ID["I-MONEY"]]
        ex = NERExample(tokens=tokens, ner_tags=tags)
        assert len(ex.tokens) == len(ex.ner_tags)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(AssertionError):
            NERExample(tokens=["A", "B"], ner_tags=[0])


class TestSyntheticDataGeneration:
    def test_generates_correct_count(self):
        examples = generate_synthetic_examples(n=50, seed=42)
        assert len(examples) == 50

    def test_examples_have_tokens(self):
        examples = generate_synthetic_examples(n=10, seed=0)
        for ex in examples:
            assert len(ex.tokens) > 0
            assert len(ex.ner_tags) == len(ex.tokens)

    def test_labels_in_valid_range(self):
        examples = generate_synthetic_examples(n=20, seed=1)
        valid_ids = set(LABEL2ID.values())
        for ex in examples:
            for tag in ex.ner_tags:
                assert tag in valid_ids, f"Invalid tag id: {tag}"

    def test_contains_entity_annotations(self):
        examples = generate_synthetic_examples(n=100, seed=42)
        # At least some examples should have non-O tags
        has_entity = any(any(t != LABEL2ID["O"] for t in ex.ner_tags) for ex in examples)
        assert has_entity

    def test_deterministic_with_seed(self):
        ex1 = generate_synthetic_examples(n=5, seed=7)
        ex2 = generate_synthetic_examples(n=5, seed=7)
        assert [e.tokens for e in ex1] == [e.tokens for e in ex2]


class TestSaveLoadJSONL:
    def test_save_and_reload(self, tmp_path):
        examples = generate_synthetic_examples(n=10, seed=42)
        jsonl_path = tmp_path / "examples.jsonl"
        save_examples_to_jsonl(examples, jsonl_path)

        from src.finetune.dataset import _load_jsonl
        loaded = _load_jsonl(jsonl_path)
        assert len(loaded) == 10
        assert loaded[0].tokens == examples[0].tokens
        assert loaded[0].ner_tags == examples[0].ner_tags


class TestFinancialNERDataset:
    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_build_synthetic_returns_dataset_dict(self, tokenizer):
        ds = FinancialNERDataset(tokenizer=tokenizer, max_length=128)
        dataset_dict = ds.build_synthetic(n_train=30, n_val=10, seed=42)
        assert "train" in dataset_dict
        assert "validation" in dataset_dict
        assert len(dataset_dict["train"]) == 30
        assert len(dataset_dict["validation"]) == 10

    def test_tokenized_dataset_has_labels(self, tokenizer):
        ds = FinancialNERDataset(tokenizer=tokenizer, max_length=128)
        dataset_dict = ds.build_synthetic(n_train=10, n_val=5, seed=0)
        example = dataset_dict["train"][0]
        assert "input_ids" in example
        assert "labels" in example
        assert "attention_mask" in example

    def test_label_ids_match_special_tokens(self, tokenizer):
        ds = FinancialNERDataset(tokenizer=tokenizer, max_length=128)
        dataset_dict = ds.build_synthetic(n_train=10, n_val=5, seed=1)
        example = dataset_dict["train"][0]
        labels = example["labels"]
        input_ids = example["input_ids"]
        # [CLS] and [SEP] tokens should have -100 label
        assert labels[0] == -100  # [CLS]


# ── Inference tests ───────────────────────────────────────────────────────────

class TestEntityClass:
    def test_to_dict(self):
        ent = Entity(text="Goldman Sachs", label="ORG", start=0, end=13, confidence=0.95)
        d = ent.to_dict()
        assert d["text"] == "Goldman Sachs"
        assert d["label"] == "ORG"
        assert d["start"] == 0
        assert d["end"] == 13
        assert 0 <= d["confidence"] <= 1


class TestAggregateEntities:
    def test_strips_bio_prefix(self):
        raw = [{"entity_group": "B-ORG", "start": 0, "end": 13, "score": 0.9, "word": "Goldman Sachs"}]
        text = "Goldman Sachs earned $10 billion."
        entities = _aggregate_entities(raw, text)
        assert any(e.label == "ORG" for e in entities)

    def test_skips_o_label(self):
        raw = [{"entity_group": "O", "start": 0, "end": 6, "score": 0.99, "word": "earned"}]
        text = "earned revenue"
        entities = _aggregate_entities(raw, text)
        assert len(entities) == 0

    def test_preserves_character_offsets(self):
        text = "Goldman Sachs reported $50 billion."
        raw = [
            {"entity_group": "ORG", "start": 0, "end": 13, "score": 0.92, "word": "Goldman Sachs"},
            {"entity_group": "MONEY", "start": 23, "end": 34, "score": 0.88, "word": "$50 billion"},
        ]
        entities = _aggregate_entities(raw, text)
        org = next((e for e in entities if e.label == "ORG"), None)
        assert org is not None
        assert org.text == "Goldman Sachs"
        assert org.start == 0
        assert org.end == 13


class TestDeduplicate:
    def test_non_overlapping_kept(self):
        entities = [
            Entity(text="Goldman Sachs", label="ORG", start=0, end=13, confidence=0.9),
            Entity(text="$50 billion", label="MONEY", start=23, end=34, confidence=0.85),
        ]
        result = _deduplicate(entities)
        assert len(result) == 2

    def test_overlapping_keeps_highest_confidence(self):
        entities = [
            Entity(text="Goldman", label="ORG", start=0, end=7, confidence=0.7),
            Entity(text="Goldman Sachs", label="ORG", start=0, end=13, confidence=0.95),
        ]
        result = _deduplicate(entities)
        # Only one entity at start=0
        starts = [e.start for e in result]
        assert starts.count(0) == 1
        kept = next(e for e in result if e.start == 0)
        assert kept.confidence == 0.95

    def test_empty_input(self):
        assert _deduplicate([]) == []

    def test_sorted_by_start(self):
        entities = [
            Entity(text="2023", label="DATE", start=30, end=34, confidence=0.9),
            Entity(text="Goldman Sachs", label="ORG", start=0, end=13, confidence=0.95),
        ]
        result = _deduplicate(entities)
        assert result[0].start < result[1].start


# ── NERInferenceEngine with tiny model ───────────────────────────────────────

class TestNERInferenceEngineWithTinyModel:
    """Test the NER engine with a tiny randomly-initialised BERT model.

    This avoids downloading large checkpoints but exercises the full inference
    code path including the HuggingFace pipeline wrapper.
    """

    @pytest.fixture
    def tiny_model_dir(self, tmp_path):
        """Create a tiny BERT model saved to disk."""
        from transformers import BertConfig, BertForTokenClassification, AutoTokenizer

        config = BertConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            num_labels=len(LABEL_LIST),
            id2label={i: l for i, l in enumerate(LABEL_LIST)},
            label2id=LABEL2ID,
        )
        model = BertForTokenClassification(config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        model_dir = tmp_path / "tiny_ner"
        model_dir.mkdir()
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        return model_dir

    def test_extract_entities_returns_list(self, tiny_model_dir):
        from src.finetune.inference import NERInferenceEngine

        engine = NERInferenceEngine(
            model_path=tiny_model_dir,
            base_model_name=str(tiny_model_dir),  # use tiny model as base
            device="cpu",
        )
        result = engine.extract_entities("Goldman Sachs earned $10 billion in Q4 2023.")
        assert isinstance(result, list)
        # Each item must be an Entity
        for ent in result:
            assert isinstance(ent, Entity)
            assert isinstance(ent.text, str)
            assert isinstance(ent.label, str)
            assert isinstance(ent.start, int)
            assert isinstance(ent.end, int)
            assert 0.0 <= ent.confidence <= 1.0

    def test_extract_entities_empty_text(self, tiny_model_dir):
        from src.finetune.inference import NERInferenceEngine

        engine = NERInferenceEngine(
            model_path=tiny_model_dir,
            base_model_name=str(tiny_model_dir),
            device="cpu",
        )
        result = engine.extract_entities("")
        assert result == []

    def test_extract_entities_batch(self, tiny_model_dir):
        from src.finetune.inference import NERInferenceEngine

        engine = NERInferenceEngine(
            model_path=tiny_model_dir,
            base_model_name=str(tiny_model_dir),
            device="cpu",
        )
        texts = [
            "Apple Inc reported $89 billion in revenue.",
            "The Fed raised rates by 25 basis points in March 2024.",
        ]
        results = engine.extract_entities_batch(texts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, list)

    def test_get_label_list(self, tiny_model_dir):
        from src.finetune.inference import NERInferenceEngine

        engine = NERInferenceEngine(
            model_path=tiny_model_dir,
            base_model_name=str(tiny_model_dir),
            device="cpu",
        )
        labels = engine.get_label_list()
        assert set(labels) == set(LABEL_LIST)
