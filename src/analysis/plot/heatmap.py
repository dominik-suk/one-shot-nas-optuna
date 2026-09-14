from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

from src.data.pamap2_labels import Pamap2ActivityType
from src.data.pamap2_loader import get_data


class HeatmapGenerator:
    def __init__(
            self,
            model: nn.Module,
            state_dict_path: Path | str | None = None,
            device: str = 'cuda',
    ):
        self.model = model
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            self.model.load_state_dict(state_dict)

        self.device = device
        self.num_classes: int = int(self.model.num_classes)
        self.activity_type: Pamap2ActivityType = self._init_activity_type()
        _, _, test_loader = get_data(activity_type=self.activity_type)
        self.data_loader: DataLoader = test_loader
        self.confusion_matrix: np.ndarray = self.generate_confusion_matrix()


    def show(self):
        self.generate_heatmap(
            destination=None,
            do_plot=True
        )

    def save(self, destination: Path | str) -> None:
        self.generate_heatmap(
            destination=Path(destination),
            do_plot=False
        )

    def generate_heatmap(
            self,
            destination: Path = None,
            do_plot: bool = True
    ):
        labels = self.activity_type.labels
        plt.figure()
        sns.heatmap(
            self.confusion_matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=False,
            cmap="Blues",
            fmt='d',
            square=True
        )
        plt.xlabel("Predicted", fontweight="bold")
        plt.ylabel("True", fontweight="bold")
        plt.title(self._get_title_label(destination), fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0, ha="right")
        plt.tight_layout()

        if destination is not None:
            destination.parent.mkdir(exist_ok=True)
            plt.savefig(destination)

        if do_plot:
            print(self.confusion_matrix)
            plt.show()

        plt.close()

    def generate_confusion_matrix(self) -> np.ndarray:
        self.model.to(self.device)
        self.model.eval()
        cm = MulticlassConfusionMatrix(num_classes=self.num_classes).to(self.device)
        with torch.no_grad():
            for features, targets in self.data_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                predictions = self.model(features)
                cm.update(predictions, targets)
        return cm.compute().cpu().numpy()

    def _init_activity_type(self) -> Pamap2ActivityType:
        if self.model.num_classes == 6:
            return Pamap2ActivityType.ADL
        if self.model.num_classes == 12:
            return Pamap2ActivityType.PROTOCOL
        return Pamap2ActivityType.ALL

    @staticmethod
    def _get_title_label(destination: Path | None) -> str:
        if destination is not None:
            title = ' '.join([part.capitalize() for part in destination.stem.split("_")][:-1])
            return f"Confusion Matrix Heatmap: {title}"
        return "Confusion Matrix Heatmap"
