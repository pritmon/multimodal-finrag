"""LoRA fine-tuning of bert-base-uncased for financial NER.

Uses PEFT (Parameter-Efficient Fine-Tuning) library with:
- LoRA adapters on query and value projection matrices
- Gradient accumulation for effective large batch training
- seqeval for precision/recall/F1 metrics
- Optional WandB integration
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from .dataset import ID2LABEL, LABEL2ID, LABEL_LIST

logger = logging.getLogger(__name__)

try:
    import evaluate
    _seqeval = evaluate.load("seqeval")
    _HAS_EVALUATE = True
except Exception:
    _HAS_EVALUATE = False
    logger.warning("evaluate/seqeval not available; metrics will be skipped")

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


@dataclass
class LoRATrainerConfig:
    """All hyperparameters and paths for the LoRA training run."""

    # Model
    base_model_name: str = "bert-base-uncased"
    output_dir: str = "./models/finrag-ner-lora"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"])

    # Training
    num_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 512
    seed: int = 42

    # Logging
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class LoRATrainer:
    """Fine-tune a BERT model for financial NER with LoRA adapters.

    Parameters
    ----------
    config:
        Training configuration.
    """

    def __init__(self, config: LoRATrainerConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("LoRATrainer using device: %s", self.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

        # Base model
        logger.info("Loading base model: %s", config.base_model_name)
        base_model = AutoModelForTokenClassification.from_pretrained(
            config.base_model_name,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)

        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "Trainable params: %d / %d (%.2f%%)",
            trainable, total, 100 * trainable / total,
        )
        self.model.print_trainable_parameters()

        # WandB
        self._wandb_run = None
        if config.wandb_project and _HAS_WANDB:
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, dataset: DatasetDict) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        dataset:
            A tokenized DatasetDict with "train" and "validation" splits.

        Returns
        -------
        dict
            Final evaluation metrics.
        """
        cfg = self.config
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8
        )

        train_loader = DataLoader(
            dataset["train"].with_format("torch"),
            batch_size=cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        eval_loader = DataLoader(
            dataset["validation"].with_format("torch"),
            batch_size=cfg.per_device_eval_batch_size,
            collate_fn=data_collator,
        )

        total_steps = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        logger.info(
            "Training: epochs=%d, steps=%d, warmup=%d",
            cfg.num_epochs, total_steps, warmup_steps,
        )

        global_step = 0
        best_f1 = 0.0
        best_metrics: dict = {}

        for epoch in range(cfg.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / cfg.gradient_accumulation_steps
                loss.backward()
                epoch_loss += outputs.loss.item()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        lr = scheduler.get_last_lr()[0]
                        logger.info(
                            "Epoch %d step %d | loss=%.4f lr=%.2e",
                            epoch + 1, global_step, avg_loss, lr,
                        )
                        if self._wandb_run:
                            wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)

                    if global_step % cfg.eval_steps == 0:
                        metrics = self.evaluate(eval_loader)
                        logger.info("Eval @ step %d: %s", global_step, metrics)
                        if self._wandb_run:
                            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)

                        if metrics.get("f1", 0) > best_f1:
                            best_f1 = metrics["f1"]
                            best_metrics = metrics
                            self._save_checkpoint("best")

            logger.info("Epoch %d complete. Avg loss: %.4f", epoch + 1, epoch_loss / len(train_loader))

        # Final evaluation and save
        final_metrics = self.evaluate(eval_loader)
        logger.info("Final metrics: %s", final_metrics)
        self.save(cfg.output_dir)

        if self._wandb_run:
            wandb.finish()

        return final_metrics if final_metrics.get("f1", 0) >= best_f1 else best_metrics

    def evaluate(self, eval_loader: DataLoader) -> dict:
        """Run evaluation on the validation set and return seqeval metrics."""
        self.model.eval()
        all_preds: list[list[str]] = []
        all_labels: list[list[str]] = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)

                label_ids = batch["labels"]
                for pred_seq, label_seq in zip(predictions, label_ids):
                    pred_labels = []
                    true_labels = []
                    for p, l in zip(pred_seq.cpu().tolist(), label_seq.cpu().tolist()):
                        if l == -100:
                            continue
                        pred_labels.append(ID2LABEL.get(p, "O"))
                        true_labels.append(ID2LABEL.get(l, "O"))
                    all_preds.append(pred_labels)
                    all_labels.append(true_labels)

        if not _HAS_EVALUATE or not all_preds:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        results = _seqeval.compute(predictions=all_preds, references=all_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def save(self, output_dir: str | Path) -> None:
        """Save LoRA adapter weights and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info("Saved LoRA adapter to %s", output_dir)

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = Path(self.config.output_dir) / f"checkpoint-{tag}"
        self.save(ckpt_dir)
