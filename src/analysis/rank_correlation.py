import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import optuna
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader

from external.build_model import construct_model, ShapeValueError
from src.data.pamap2_loader import get_data
from src.models.supernet import Supernet
from src.models.train_model import train, evaluate
from src.nas.supernet_trainer import SupernetTrainingWrapper
from src.paths import SUPERNET_PATH, RANKING_CORRELATION_DATA_PATH
from src.utils.yaml_io import load_pamap2_search_space


@dataclass
class RankCorrelation:
    timestamp: str
    supernet: list[float]
    standalone: list[float]
    spearman_corr: float
    spearman_p: float
    kendall_corr: float
    kendall_p: float

    def write_to_csv(self, path: Path = RANKING_CORRELATION_DATA_PATH) -> None:
        rounded_path = path.parent / f"{path.stem}_rounded_values{path.suffix}"
        os.makedirs(path.parent, exist_ok=True)

        class_dict = asdict(self)
        class_dict['supernet'] = self._to_string(class_dict['supernet'])
        class_dict['standalone'] = self._to_string(class_dict['standalone'])

        pd.DataFrame([class_dict]).to_csv(
            path,
            mode='a',
            index=False,
            header=not path.is_file()
        )

        rounded_dict = {
            'Timestamp': self.timestamp,
            'Supernet Architectures': self._to_string(self.supernet, do_round=True),
            'Standalone Architectures': self._to_string(self.standalone, do_round=True),
            'Spearman Corr': round(self.spearman_corr, 2),
            'Spearman p': round(self.spearman_p, 4),
            'Kendall Corr': round(self.kendall_corr, 2),
            'Kendall p': round(self.kendall_p, 4),
        }

        pd.DataFrame([rounded_dict]).to_csv(
            rounded_path,
            mode='a',
            index=False,
            header=not rounded_path.is_file()
        )

    def print(self):
        print(f"Supernet {len(self.supernet)} Ranked Accuracies: {[f"{acc:.2f}" for acc in self.supernet]}")
        print(f"Standalone {len(self.supernet)} Ranked Accuracies: {[f"{acc:.2f}" for acc in self.standalone]}")
        print(f"Spearman Correlation: {self.spearman_corr:.2f}")
        print(f"Spearman p: {self.spearman_p:.4f}")
        print(f"Kendall Correlation: {self.kendall_corr:.2f}")
        print(f"Kendall p: {self.kendall_p:.4f}")

    @staticmethod
    def _to_string(values: list, do_round: bool = False):
        return ", ".join([f"A{i + 1}: {value if not do_round else round(value, 2)}" for i, value in enumerate(values)])


def sample_n_random_architectures(supernet: Supernet, n: int = 25) -> list:
    sampled_architectures = []

    wrapped_supernet = SupernetTrainingWrapper(
        supernet=supernet,
    )

    while len(sampled_architectures) < n:
        architecture = wrapped_supernet.take_one_sample()

        try:
            _ = construct_model(
                architecture,
                in_dim=supernet.search_space['input'],
                out_dim=supernet.search_space['output']
            )
            sampled_architectures.append(architecture)

        except ShapeValueError:
            continue

    return sampled_architectures


def conduct_rank_correlation(n: int, device: str = 'cuda'):
    training_loader, validation_loader, test_loader = get_data()
    search_space = load_pamap2_search_space()

    supernet = Supernet(search_space=search_space).to(device)
    supernet_state_dict = torch.load(f=SUPERNET_PATH, map_location=device)
    supernet.load_state_dict(supernet_state_dict)
    sampled_architectures = sample_n_random_architectures(supernet, n)

    supernet_acc_values = evaluate_with_supernet(
        supernet=supernet,
        sampled_architectures=sampled_architectures,
        validation_loader=validation_loader
    )

    standalone_acc_values = evaluate_with_standalone_training(
        sampled_architectures=sampled_architectures,
        data_loaders=(training_loader, validation_loader, test_loader),
        search_space=search_space,
        device=device
    )

    spearman_corr, spearman_p = spearmanr(supernet_acc_values, standalone_acc_values)
    kendall_corr, kendall_p = kendalltau(supernet_acc_values, standalone_acc_values)

    return RankCorrelation(
        timestamp=datetime.now().strftime("%d.%m.%Y-%H:%M:%S"),
        supernet=supernet_acc_values,
        standalone=standalone_acc_values,
        spearman_corr=spearman_corr,
        spearman_p=spearman_p,
        kendall_corr=kendall_corr,
        kendall_p=kendall_p,
    )


def evaluate_with_supernet(supernet: Supernet, sampled_architectures: list, validation_loader, device: str = 'cuda'):
    supernet_acc_values = []
    for i, architecture in enumerate(sampled_architectures):
        wrapped_supernet = SupernetTrainingWrapper(
            supernet=supernet,
            fixed_subnet_path=architecture,
        )
        summary = evaluate(
            model=wrapped_supernet,
            data_loader=validation_loader,
            device=device
        )
        print(f"Supernet Architecture Nr. {i + 1}: {summary.accuracy:.2f} % Accuracy")
        supernet_acc_values.append(summary.accuracy)

    return supernet_acc_values


def evaluate_with_standalone_training(sampled_architectures: list, data_loaders: tuple[DataLoader, DataLoader, DataLoader], search_space: dict, device: str = 'cuda'):
    standalone_values = []

    for i, architecture in enumerate(sampled_architectures):
        standalone_model = construct_model(
            architecture,
            in_dim=search_space['input'],
            out_dim=search_space['output']
        ).to(device)

        summary = train(
            model=standalone_model,
            epochs=50,
            n_proxy_epochs=15,
            load_best_weights=True,
            retraining_best_model=False,
            data_loaders=data_loaders,
            device=device,
            logger=None,
        )

        print(f"Standalone Architecture Nr. {i + 1}: {summary.accuracy:.2f} % Accuracy")
        standalone_values.append(summary.accuracy)

    return standalone_values


if __name__ == "__main__":
    time_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    rank_correlation = conduct_rank_correlation(n=100, device='cuda')
    rank_correlation.print()
    rank_correlation.write_to_csv()

    time_end = time.time()
    minutes, seconds = divmod(time_end - time_start, 60)
    print(f"Time: {int(minutes)}m {int(seconds)}s")
