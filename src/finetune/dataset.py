"""Financial NER dataset in HuggingFace format.

Labels: O, B-ORG, I-ORG, B-MONEY, I-MONEY, B-DATE, I-DATE, B-PERCENT, I-PERCENT

Supports:
- Loading from JSONL annotation files
- Synthetic training data generation for bootstrapping
- Tokenization with subword-label alignment (WordPiece / BPE)
- HuggingFace Dataset + DatasetDict construction
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class NERLabel(str, Enum):
    O = "O"
    B_ORG = "B-ORG"
    I_ORG = "I-ORG"
    B_MONEY = "B-MONEY"
    I_MONEY = "I-MONEY"
    B_DATE = "B-DATE"
    I_DATE = "I-DATE"
    B_PERCENT = "B-PERCENT"
    I_PERCENT = "I-PERCENT"


LABEL_LIST = [label.value for label in NERLabel]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


@dataclass
class NERExample:
    """A single NER training example with word-level token + label alignment."""

    tokens: list[str]
    ner_tags: list[int]  # integer IDs from LABEL2ID
    doc_id: Optional[str] = None

    def __post_init__(self) -> None:
        assert len(self.tokens) == len(self.ner_tags), (
            f"tokens ({len(self.tokens)}) and ner_tags ({len(self.ner_tags)}) must have equal length"
        )


# ── Synthetic data templates ──────────────────────────────────────────────────

_ORGS = [
    "Goldman Sachs", "JPMorgan Chase", "Morgan Stanley", "BlackRock",
    "Vanguard Group", "Fidelity Investments", "Bank of America", "Citigroup",
    "Wells Fargo", "UBS Group", "Deutsche Bank", "Barclays", "HSBC Holdings",
    "Apple Inc", "Microsoft Corporation", "Alphabet Inc", "Amazon.com Inc",
    "Tesla Inc", "Meta Platforms", "Berkshire Hathaway",
]

_MONEY_TEMPLATES = [
    ("{amount} million", ["B-MONEY", "I-MONEY"]),
    ("{amount} billion", ["B-MONEY", "I-MONEY"]),
    ("${amount}", ["B-MONEY"]),
    ("${amount} million", ["B-MONEY", "I-MONEY"]),
    ("${amount} billion", ["B-MONEY", "I-MONEY"]),
    ("{amount} trillion dollars", ["B-MONEY", "I-MONEY", "I-MONEY"]),
]

_PERCENT_TEMPLATES = [
    ("{pct}%", ["B-PERCENT"]),
    ("{pct} percent", ["B-PERCENT", "I-PERCENT"]),
    ("{pct} basis points", ["B-PERCENT", "I-PERCENT", "I-PERCENT"]),
]

_DATE_TEMPLATES = [
    ("Q{q} {year}", ["B-DATE", "I-DATE"]),
    ("{month} {year}", ["B-DATE", "I-DATE"]),
    ("fiscal year {year}", ["B-DATE", "I-DATE", "I-DATE"]),
    ("FY{year}", ["B-DATE"]),
    ("{month} {day}, {year}", ["B-DATE", "I-DATE", "I-DATE"]),
]

_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]

_SENTENCE_PATTERNS = [
    "{org} reported revenue of {money} for {date}, representing a {pct} increase year-over-year.",
    "The acquisition of {org} was valued at {money}, completed in {date}.",
    "{org} announced a {pct} dividend increase, bringing the annual payout to {money}.",
    "Net income at {org} fell {pct} to {money} in {date} amid rising interest rates.",
    "Analysts at {org} raised their price target by {pct}, citing strong earnings of {money}.",
    "Operating margins for {org} expanded {pct} to reach {money} in {date}.",
    "{org} issued {money} in corporate bonds with a maturity date in {date}.",
    "The merger between {org} and its rival was completed in {date} for {money}.",
    "{org} returned {money} to shareholders through buybacks in {date}, a {pct} increase.",
    "Total assets under management at {org} reached {money}, up {pct} from {date}.",
]


def _random_amount() -> str:
    return str(round(random.uniform(0.5, 999.9), 1))


def _random_pct() -> str:
    return str(round(random.uniform(0.1, 45.0), 1))


def _random_date_tokens() -> tuple[list[str], list[str]]:
    template, labels = random.choice(_DATE_TEMPLATES)
    year = str(random.randint(2015, 2024))
    month = random.choice(_MONTHS)
    day = str(random.randint(1, 28))
    q = str(random.randint(1, 4))
    text = template.format(year=year, month=month, day=day, q=q)
    tokens = text.split()
    return tokens, labels[: len(tokens)]  # align label count to token count


def _random_money_tokens() -> tuple[list[str], list[str]]:
    template, labels = random.choice(_MONEY_TEMPLATES)
    text = template.format(amount=_random_amount())
    tokens = text.split()
    return tokens, labels[: len(tokens)]


def _random_pct_tokens() -> tuple[list[str], list[str]]:
    template, labels = random.choice(_PERCENT_TEMPLATES)
    text = template.format(pct=_random_pct())
    tokens = text.split()
    return tokens, labels[: len(tokens)]


def generate_synthetic_examples(n: int = 1000, seed: int = 42) -> list[NERExample]:
    """Generate synthetic financial NER training examples.

    Each example is a sentence with annotated ORG, MONEY, DATE, PERCENT spans.
    """
    random.seed(seed)
    examples: list[NERExample] = []

    for i in range(n):
        pattern = random.choice(_SENTENCE_PATTERNS)
        org = random.choice(_ORGS)
        money_toks, money_labels = _random_money_tokens()
        pct_toks, pct_labels = _random_pct_tokens()
        date_toks, date_labels = _random_date_tokens()

        # Build the sentence token-by-token
        org_tokens = org.split()
        org_labels = ["B-ORG"] + ["I-ORG"] * (len(org_tokens) - 1)

        money_str = " ".join(money_toks)
        pct_str = " ".join(pct_toks)
        date_str = " ".join(date_toks)

        sentence = pattern.format(
            org=org, money=money_str, pct=pct_str, date=date_str
        )

        tokens, ner_tags = _annotate_sentence(
            sentence,
            {
                org: org_labels,
                money_str: money_labels,
                pct_str: pct_labels,
                date_str: date_labels,
            },
        )

        examples.append(
            NERExample(tokens=tokens, ner_tags=ner_tags, doc_id=f"synthetic_{i}")
        )

    logger.info("Generated %d synthetic NER examples", n)
    return examples


def _annotate_sentence(
    sentence: str,
    span_labels: dict[str, list[str]],
) -> tuple[list[str], list[int]]:
    """Tokenize a sentence and assign BIO labels based on known spans."""
    words = sentence.split()
    label_ids = [LABEL2ID["O"]] * len(words)

    for span_text, span_bio in span_labels.items():
        span_words = span_text.split()
        for i in range(len(words) - len(span_words) + 1):
            if words[i : i + len(span_words)] == span_words:
                for j, bio in enumerate(span_bio):
                    if i + j < len(label_ids):
                        label_ids[i + j] = LABEL2ID.get(bio, LABEL2ID["O"])
                break

    return words, label_ids


# ── Dataset class ─────────────────────────────────────────────────────────────

class FinancialNERDataset:
    """Build and manage the financial NER HuggingFace dataset.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer (must be fast for word-level alignment).
    max_length:
        Maximum subword sequence length after tokenization.
    label_all_tokens:
        If True, assign the entity label to all subword tokens of a word;
        if False (default), only the first subword token gets the entity
        label and the rest are masked with -100.
    """

    LABEL_LIST = LABEL_LIST
    LABEL2ID = LABEL2ID
    ID2LABEL = ID2LABEL

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        label_all_tokens: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens

    def from_examples(
        self,
        train_examples: list[NERExample],
        val_examples: Optional[list[NERExample]] = None,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """Build a tokenized DatasetDict from NERExample lists.

        If ``val_examples`` is None, a ``val_split`` fraction is split from
        ``train_examples``.
        """
        if val_examples is None:
            random.seed(seed)
            shuffled = list(train_examples)
            random.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * (1 - val_split)))
            train_examples = shuffled[:split_idx]
            val_examples = shuffled[split_idx:]

        train_ds = self._examples_to_hf_dataset(train_examples)
        val_ds = self._examples_to_hf_dataset(val_examples)

        tokenized_train = train_ds.map(self._tokenize_and_align, batched=True)
        tokenized_val = val_ds.map(self._tokenize_and_align, batched=True)

        return DatasetDict({"train": tokenized_train, "validation": tokenized_val})

    def from_jsonl(self, path: str | Path, **kwargs) -> DatasetDict:
        """Load annotated examples from a JSONL file and build the dataset.

        Each line must be JSON with ``tokens`` (list[str]) and
        ``ner_tags`` (list[int] or list[str]) fields.
        """
        examples = _load_jsonl(path)
        return self.from_examples(examples, **kwargs)

    def build_synthetic(self, n_train: int = 2000, n_val: int = 300, seed: int = 42) -> DatasetDict:
        """Generate synthetic data and return a ready-to-train DatasetDict."""
        all_examples = generate_synthetic_examples(n_train + n_val, seed=seed)
        train_examples = all_examples[:n_train]
        val_examples = all_examples[n_train:]
        return self.from_examples(train_examples, val_examples)

    # ── Private ───────────────────────────────────────────────────────────────

    def _examples_to_hf_dataset(self, examples: list[NERExample]) -> Dataset:
        data = {
            "tokens": [e.tokens for e in examples],
            "ner_tags": [e.ner_tags for e in examples],
            "doc_id": [e.doc_id or "" for e in examples],
        }
        return Dataset.from_dict(data)

    def _tokenize_and_align(self, batch: dict) -> dict:
        """Tokenize a batch and align NER labels to subword tokens."""
        tokenized = self.tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
        )
        labels_batch = []
        for i, word_ids in enumerate(tokenized.word_ids(batch_index=j) for j in range(len(batch["tokens"]))):
            word_labels = batch["ner_tags"][i]
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(word_labels[word_idx])
                else:
                    if self.label_all_tokens:
                        label_ids.append(word_labels[word_idx])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            labels_batch.append(label_ids)

        tokenized["labels"] = labels_batch
        return tokenized


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_jsonl(path: str | Path) -> list[NERExample]:
    examples: list[NERExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = obj["tokens"]
            raw_tags = obj["ner_tags"]
            if raw_tags and isinstance(raw_tags[0], str):
                ner_tags = [LABEL2ID.get(t, 0) for t in raw_tags]
            else:
                ner_tags = [int(t) for t in raw_tags]
            examples.append(
                NERExample(tokens=tokens, ner_tags=ner_tags, doc_id=obj.get("doc_id"))
            )
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def save_examples_to_jsonl(examples: list[NERExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(
                json.dumps(
                    {"tokens": ex.tokens, "ner_tags": ex.ner_tags, "doc_id": ex.doc_id}
                )
                + "\n"
            )
    logger.info("Saved %d examples to %s", len(examples), path)
