import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.paths import TRAIN_HISTORY_DIR


@dataclass
class ModelSummary:
    loss: float | None
    accuracy: float | None
    f1_score: float | None
    history: dict[str, list[float | int]] | None = None

    def print(self):
        print(
            "Model Summary:\n"
            f"Loss: {self.loss:.4f}\n"
            f"Accuracy: {self.accuracy:.2f} %\n"
            f"F1 Score: {self.f1_score:.2f} %\n"
        )


class HistoryTracker:
    def __init__(self):
        self.history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    def update(self, epoch: int, train_summary: ModelSummary, val_summary: ModelSummary):
        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(train_summary.loss)
        self.history["val_loss"].append(val_summary.loss)
        self.history["train_acc"].append(train_summary.accuracy)
        self.history["val_acc"].append(val_summary.accuracy)

    def to_dict(self) -> dict[str, list[float | int]]:
        return self.history
