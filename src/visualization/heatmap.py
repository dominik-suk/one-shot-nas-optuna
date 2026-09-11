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
from src.models.fixed_architecture import HumanActivityClassifier
from src.paths import HEATMAPS_DIR, SAMPLED_ADL_MODEL_PATH, SAMPLED_PROTOCOL_MODEL_PATH


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
            destination=destination,
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
            annot=True,
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


def main():
    protocol_heatmap = HeatmapGenerator(
        model=HumanActivityClassifier(num_classes=12),
        state_dict_path=SAMPLED_PROTOCOL_MODEL_PATH,
    )
    protocol_heatmap.show()
    protocol_heatmap.save(HEATMAPS_DIR / f"{SAMPLED_PROTOCOL_MODEL_PATH.stem}_heatmap.png")

    adl_heatmap = HeatmapGenerator(
        model=HumanActivityClassifier(num_classes=6),
        state_dict_path=SAMPLED_ADL_MODEL_PATH,
    )
    adl_heatmap.show()
    adl_heatmap.save(HEATMAPS_DIR / f"{SAMPLED_ADL_MODEL_PATH.stem}_heatmap.png")


    all_heatmap = HeatmapGenerator(
        model=HumanActivityClassifier(num_classes=18),
        state_dict_path=Path("/home/dominik/Documents/Uni/Abschlussarbeit/Code/one-shot-nas-optuna/models/samples/fixed_arch_ALL_acc_53.pth"),
    )
    all_heatmap.show()
    all_heatmap.save(Path("/home/dominik/Documents/Uni/Abschlussarbeit/Code/one-shot-nas-optuna/plots/heatmaps/fixed_arch_ALL_acc_53_heatmap.png"))

if __name__ == "__main__":
    main()