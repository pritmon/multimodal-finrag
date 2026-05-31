"""Inference engine for the LoRA-fine-tuned financial NER model.

HOW IT WORKS (simple analogy):
  This is like a UiPath activity that takes text as input and returns
  a list of labelled entities found in that text.

  The model was fine-tuned with LoRA (Low-Rank Adaptation) — a technique
  that trains only a small fraction of the model's parameters (~1-5%)
  instead of all of them. This makes fine-tuning fast and cheap.

  At inference time:
    1. Load the base BERT model (bert-base-uncased)
    2. Apply the LoRA adapter (small weight diff files)
    3. Merge the adapter into the base model (faster inference)
    4. Run input text through the merged model → BIO token labels
    5. Aggregate consecutive tokens with the same entity label
    6. Return Entity objects with character positions + confidence

  BIO labelling:
    B-ORG    = beginning of an ORG entity
    I-ORG    = continuation of an ORG entity
    O        = not an entity
    Example: "Goldman | Sachs | reported" → B-ORG | I-ORG | O

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
from peft import PeftModel  # PEFT = Parameter-Efficient Fine-Tuning library
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from .dataset import ID2LABEL, LABEL2ID, LABEL_LIST

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A single named entity detected in text.

    Contains everything needed to locate and understand the entity:
    - What it says (text)
    - What type it is (label)
    - Where it is in the original string (start, end)
    - How confident the model is (confidence)
    """

    text: str           # the exact substring matched in the input (e.g. "Goldman Sachs")
    label: str          # entity type: "ORG", "MONEY", "DATE", or "PERCENT"
    start: int          # character offset where the entity starts (inclusive)
    end: int            # character offset where the entity ends (exclusive)
    confidence: float   # average softmax probability across subword tokens (0.0 to 1.0)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for API responses."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
        }


class NERInferenceEngine:
    """Load a LoRA-fine-tuned NER model and extract financial named entities.

    Merges the LoRA adapter into the base BERT model at load time,
    which makes inference faster (single model forward pass instead of two).

    Parameters
    ----------
    model_path:
        Directory containing the saved LoRA adapter (from ``LoRATrainer.save()``).
        Should contain: adapter_config.json, adapter_model.safetensors, tokenizer files.
    base_model_name:
        HuggingFace model ID for the base model (must match what was used for training).
    device:
        PyTorch device string. Auto-detects CUDA if None.
    batch_size:
        Batch size for ``extract_entities_batch()``.
    max_length:
        Maximum input sequence length in subword tokens (512 for BERT).
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
        # Auto-detect CUDA GPU; fall back to CPU
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)

        # These are set in _load_model()
        self.tokenizer: AutoTokenizer = None  # type: ignore[assignment]
        self.model: PeftModel = None          # type: ignore[assignment]
        self._pipeline = None                  # HuggingFace pipeline object

        self._load_model()  # load model immediately at construction

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract named entities from a single text string.

        Returns a deduplicated, merged list of Entity objects sorted by
        start character offset (earliest in the text first).

        Example:
          entities = engine.extract_entities("Apple reported $5B revenue in Q3 2024")
          # → [Entity("Apple", "ORG", 0, 5, 0.98),
          #     Entity("$5B", "MONEY", 16, 19, 0.95),
          #     Entity("Q3 2024", "DATE", 31, 38, 0.91)]
        """
        if not text.strip():
            return []  # skip empty input — nothing to extract

        # Run the HuggingFace pipeline (tokenise → model forward pass → decode)
        raw = self._pipeline(text)
        # Aggregate BIO tokens into Entity objects and deduplicate overlaps
        return _aggregate_entities(raw, text)

    def extract_entities_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from a list of texts.

        Processes texts in batches for efficiency.
        Returns a list of Entity lists — one per input text (parallel structure).

        Example:
          results = engine.extract_entities_batch(["text1", "text2"])
          results[0]  # entities from text1
          results[1]  # entities from text2
        """
        results: list[list[Entity]] = []
        # Process in batches of batch_size to avoid OOM errors on large inputs
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for text in batch:
                results.append(self.extract_entities(text))
        return results

    def get_label_list(self) -> list[str]:
        """Return the list of all entity label strings the model can predict."""
        return LABEL_LIST

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the base model + LoRA adapter and create the HuggingFace pipeline.

        Steps:
          1. Load the tokenizer (from adapter dir or base model)
          2. Load the base BERT model with NER classification head
          3. If adapter_config.json exists → apply LoRA adapter weights
          4. Merge adapter into base model (merge_and_unload) for faster inference
          5. Create a HuggingFace token-classification pipeline

        merge_and_unload() fuses the LoRA weight delta into the original weights,
        producing a single model with no runtime overhead from the adapter math.
        """
        logger.info("Loading NER model from %s", self.model_path)

        # Load tokenizer — try the adapter directory first, fall back to base model
        # (tokenizer is saved alongside the adapter by LoRATrainer.save())
        tok_path = (
            self.model_path
            if (self.model_path / "tokenizer_config.json").exists()
            else self.base_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))

        # Load the base BERT model with a token classification head
        # (9 output classes: O + 4 entity types × 2 BIO prefixes)
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,     # {0: "O", 1: "B-ORG", ...}
            label2id=LABEL2ID,     # {"O": 0, "B-ORG": 1, ...}
            ignore_mismatched_sizes=True,  # needed when fine-tuning changes the classifier head
        )

        # Apply LoRA adapter if the adapter config file is present
        if (self.model_path / "adapter_config.json").exists():
            logger.info("Loading LoRA adapter from %s", self.model_path)
            # Wrap the base model with the LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            # Merge the adapter weights into the base model — faster inference
            # (no longer need separate base + adapter; weights are fused)
            self.model = self.model.merge_and_unload()
        else:
            # No adapter found — use base model only (no fine-tuning applied)
            logger.warning("No adapter_config.json found; using base model only")
            self.model = base_model  # type: ignore[assignment]

        # Move model to the target device (GPU if available, else CPU)
        self.model.to(self.device)
        self.model.eval()  # set to evaluation mode (disables dropout, etc.)

        # Create a HuggingFace NER pipeline — handles tokenisation, model call, decoding
        # aggregation_strategy="first": use the first subword token's label for each word
        # device=0 for GPU, device=-1 for CPU
        device_id = 0 if self.device_str == "cuda" else -1
        self._pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",  # aggregate BIO tokens per word
            device=device_id,
        )
        logger.info("NER inference engine ready on %s", self.device_str)


# ── Post-processing Utilities ─────────────────────────────────────────────────

def _aggregate_entities(raw_entities: list[dict], original_text: str) -> list[Entity]:
    """Convert HuggingFace pipeline output to Entity objects.

    The pipeline returns a list of dicts like:
      {"entity_group": "ORG", "score": 0.98, "start": 0, "end": 13, "word": "Goldman Sachs"}

    We:
      1. Strip BIO prefixes (B-ORG → ORG, I-ORG → ORG)
      2. Skip "O" (non-entity) tokens
      3. Extract the exact text from the original string using start/end offsets
      4. Build Entity objects
      5. Deduplicate overlapping entities (keep the one with higher confidence)

    We use the original_text for entity text because the pipeline's "word" field
    may have tokenization artifacts (##suffix tokens, etc.).
    """
    entities: list[Entity] = []

    for ent in raw_entities:
        # "entity_group" is used by aggregation_strategy="first"
        # "entity" is used without aggregation — check both
        raw_label: str = ent.get("entity_group", ent.get("entity", "O"))

        # Strip BIO prefix: "B-ORG" → "ORG", "I-MONEY" → "MONEY"
        # lstrip("BI-") removes all leading B, I, or - characters
        label = raw_label.lstrip("BI-")
        if not label or label == "O":
            continue  # skip non-entity tokens

        start = ent.get("start", 0)
        end = ent.get("end", 0)
        # Extract the exact matched text from the original string
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

    # Remove overlapping entities — keep the highest confidence one
    return _deduplicate(entities)


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Remove overlapping entities, keeping the highest-confidence one.

    For example, if two entities span the same characters:
      Entity("Goldman Sachs Group", "ORG", 0, 19, 0.95)
      Entity("Goldman", "ORG", 0, 7, 0.82)

    We keep the one that starts earliest; if they start at the same position,
    we keep the one with higher confidence.

    Algorithm:
      1. Sort by start position (earlier first), then by confidence (higher first)
      2. Walk through sorted entities, only keeping ones that don't overlap
         with the last entity we kept (non-overlapping = starts after last end)
    """
    if not entities:
        return entities

    # Sort: primarily by start position (ascending), secondarily by confidence (descending)
    sorted_ents = sorted(entities, key=lambda e: (e.start, -e.confidence))
    result: list[Entity] = []
    last_end = -1  # tracks where the last kept entity ended

    for ent in sorted_ents:
        # Only keep this entity if it starts AFTER the previous entity ended
        if ent.start >= last_end:
            result.append(ent)
            last_end = ent.end  # update the "end of last kept entity" marker

    # Final sort by start position for clean output
    return sorted(result, key=lambda e: e.start)
