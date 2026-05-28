#!/usr/bin/env python3
"""CLI: Run LoRA fine-tuning on the financial NER dataset.

Usage examples:
    # Train on synthetic data
    python scripts/train_lora.py --output-dir ./models/finrag-ner-lora

    # Train on real annotated data
    python scripts/train_lora.py \\
        --data-path /data/financial_ner_train.jsonl \\
        --output-dir ./models/finrag-ner-lora \\
        --epochs 10 --batch-size 32 --lr 2e-4 --lora-r 16

    # Quick smoke test
    python scripts/train_lora.py --synthetic --max-train-samples 200 --epochs 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.finetune.dataset import FinancialNERDataset, generate_synthetic_examples, save_examples_to_jsonl
from src.finetune.lora_trainer import LoRATrainer, LoRATrainerConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("train_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a LoRA NER model for financial entity extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-path",
        default=None,
        help="Path to a JSONL annotation file",
    )
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Generate synthetic training data (no annotation file needed)",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap on training samples (useful for smoke tests)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (when --data-path is used)",
    )

    # Model
    parser.add_argument("--base-model", default="bert-base-uncased", help="Base HF model ID")
    parser.add_argument(
        "--output-dir",
        default="./models/finrag-ner-lora",
        help="Directory to save the trained LoRA adapter",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["query", "value"],
        help="BERT sub-modules to apply LoRA to",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device eval batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--wandb-project", default=None, help="WandB project name (optional)")
    parser.add_argument("--wandb-run-name", default=None, help="WandB run name (optional)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    logger.info("=== FinRAG LoRA NER Trainer ===")
    logger.info("Base model:   %s", args.base_model)
    logger.info("Output dir:   %s", args.output_dir)
    logger.info("LoRA rank:    %d", args.lora_r)
    logger.info("Epochs:       %d", args.epochs)
    logger.info("Learning rate: %g", args.lr)

    # ── Build trainer (loads model) ───────────────────────────────────────────
    config = LoRATrainerConfig(
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    trainer = LoRATrainer(config=config)

    # ── Build dataset ─────────────────────────────────────────────────────────
    ner_dataset = FinancialNERDataset(
        tokenizer=trainer.tokenizer,
        max_length=args.max_length,
    )

    if args.data_path:
        logger.info("Loading annotation data from %s", args.data_path)
        dataset = ner_dataset.from_jsonl(args.data_path, val_split=args.val_split, seed=args.seed)
    else:
        logger.info("Generating synthetic training data")
        n_total = (args.max_train_samples or 2000) + 300
        dataset = ner_dataset.build_synthetic(
            n_train=args.max_train_samples or 2000, n_val=300, seed=args.seed
        )

    # Cap samples if requested
    if args.max_train_samples and len(dataset["train"]) > args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
        logger.info("Capped training set to %d samples", args.max_train_samples)

    logger.info(
        "Dataset: %d train, %d val",
        len(dataset["train"]),
        len(dataset["validation"]),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    metrics = trainer.train(dataset)

    logger.info("=== Training Complete ===")
    logger.info("Final metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)
    logger.info("Model saved to: %s", args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
