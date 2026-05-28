"""Inference engine for the LoRA-fine-tuned financial NER model.

Loads base model + LoRA adapter with PEFT and exposes:
- ``extract_entities(text)`` for single-string extraction
- ``extract_entities_batch(texts)`` for batch processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from .dataset import ID2LABEL, LABEL2ID, LABEL_LIST

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A named entity extracted from text."""

    text: str
    label: str          # e.g. "ORG", "MONEY", "DATE", "PERCENT"
    start: int          # character start offset in the original string
    end: int            # character end offset (exclusive)
    confidence: float   # mean softmax probability across subword tokens

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
        }


class NERInferenceEngine:
    """Load a LoRA-fine-tuned NER model and extract financial entities.

    Parameters
    ----------
    model_path:
        Directory containing the saved LoRA adapter (from ``LoRATrainer.save()``).
    base_model_name:
        HuggingFace model ID for the base model; defaults to bert-base-uncased.
    device:
        PyTorch device string. Auto-detects CUDA if None.
    batch_size:
        Batch size for ``extract_entities_batch()``.
    max_length:
        Maximum input sequence length in subword tokens.
    """

    def __init__(
        self,
        model_path: str | Path,
        base_model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)

        self.tokenizer: AutoTokenizer = None  # type: ignore[assignment]
        self.model: PeftModel = None  # type: ignore[assignment]
        self._pipeline = None

        self._load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities from a single text string.

        Returns a deduplicated, merged list of Entity objects sorted by start
        character offset.
        """
        if not text.strip():
            return []

        raw = self._pipeline(text)
        return _aggregate_entities(raw, text)

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from a list of texts.

        Parameters
        ----------
        texts:
            Input strings (may be long; will be chunked if necessary).

        Returns
        -------
        list[list[Entity]]
            Parallel list of entity lists, one per input text.
        """
        results: list[list[Entity]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for text in batch:
                results.append(self.extract_entities(text))
        return results

    def get_label_list(self) -> list[str]:
        return LABEL_LIST

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        logger.info("Loading NER model from %s", self.model_path)

        # Load tokenizer (saved alongside the adapter)
        tok_path = self.model_path if (self.model_path / "tokenizer_config.json").exists() else self.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))

        # Load base model
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        # Apply LoRA adapter if adapter_config.json is present
        if (self.model_path / "adapter_config.json").exists():
            logger.info("Loading LoRA adapter from %s", self.model_path)
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            self.model = self.model.merge_and_unload()  # merge for faster inference
        else:
            logger.warning("No adapter_config.json found; using base model only")
            self.model = base_model  # type: ignore[assignment]

        self.model.to(self.device)
        self.model.eval()

        device_id = 0 if self.device_str == "cuda" else -1
        self._pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
            device=device_id,
        )
        logger.info("NER inference engine ready on %s", self.device_str)


# ── Post-processing ───────────────────────────────────────────────────────────

def _aggregate_entities(raw_entities: list[dict], original_text: str) -> list[Entity]:
    """Convert HuggingFace pipeline output to Entity objects.

    The pipeline already handles BIO aggregation with ``aggregation_strategy="first"``.
    We strip leading 'B-' / 'I-' prefixes and map back to character positions.
    """
    entities: list[Entity] = []

    for ent in raw_entities:
        raw_label: str = ent.get("entity_group", ent.get("entity", "O"))
        # Strip BIO prefix if present
        label = raw_label.lstrip("BI-")
        if not label or label == "O":
            continue

        start = ent.get("start", 0)
        end = ent.get("end", 0)
        word = original_text[start:end]
        score = float(ent.get("score", 0.0))

        entities.append(
            Entity(
                text=word,
                label=label,
                start=start,
                end=end,
                confidence=score,
            )
        )

    # Deduplicate overlapping entities (keep highest confidence)
    return _deduplicate(entities)


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Remove overlapping entities, keeping the highest-confidence one."""
    if not entities:
        return entities

    sorted_ents = sorted(entities, key=lambda e: (e.start, -e.confidence))
    result: list[Entity] = []
    last_end = -1

    for ent in sorted_ents:
        if ent.start >= last_end:
            result.append(ent)
            last_end = ent.end

    return sorted(result, key=lambda e: e.start)
