"""Fine-tuning subpackage: financial NER dataset, LoRA training, and inference."""

from .dataset import FinancialNERDataset, NERLabel
from .inference import Entity, NERInferenceEngine
from .lora_trainer import LoRATrainer, LoRATrainerConfig

__all__ = [
    "FinancialNERDataset",
    "NERLabel",
    "LoRATrainer",
    "LoRATrainerConfig",
    "NERInferenceEngine",
    "Entity",
]
