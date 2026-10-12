import json
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

from external.build_model import construct_model
from src.data.pamap2_loader import load_pamap2_dataset
from src.models.subnet import Subnet, SubnetPathGenerator
from src.models.supernet import Supernet
from src.paths import SUPERNET_PATH, RANKING_CORRELATION_DATA_PATH, LOGS_DIR
from src.training.evaluation import evaluate
from src.training.model_trainer import ModelTrainer
from src.utils.yaml_io import load_pamap2_search_space

CHECKPOINT_CSV = LOGS_DIR / "rank_correlation_checkpoint.csv"
ARCHITECTURES_JSON = LOGS_DIR / "rank_correlation_architectures.json"

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
        os.makedirs(path.parent, exist_ok=True)
        pd.DataFrame([asdict(self)]).to_csv(
            path,
            mode='a',
            index=False,
            header=not path.is_file()
        )

    def print(self):
        print(f"Supernet {len(self.supernet)} Ranked Accuracies: {[f"{acc:.2f}" for acc in self.supernet]}")
        print(f"Standalone {len(self.supernet)} Ranked Accuracies: {[f"{acc:.2f}" for acc in self.standalone]}")
        print(f"Spearman Correlation: {self.spearman_corr:.2f}")
        print(f"Spearman p: {self.spearman_p:.4f}")
        print(f"Kendall Correlation: {self.kendall_corr:.2f}")
        print(f"Kendall p: {self.kendall_p:.4f}")


def sample_n_random_architectures(supernet: Supernet, n: int = 500) -> list:
    path_generator = SubnetPathGenerator(
        search_space=supernet.search_space,
    )

    return [path_generator.generate() for _ in range(n)]

def get_or_create_architectures(supernet: Supernet, n: int) -> list:
    if ARCHITECTURES_JSON.exists():
        with open(ARCHITECTURES_JSON) as f:
            architecture_samples = json.load(f)

        if len(architecture_samples) >= n:
            return architecture_samples[:n]

    path_generator = SubnetPathGenerator(supernet.search_space)
    architecture_samples = [path_generator.generate() for _ in range(n)]

    os.makedirs(ARCHITECTURES_JSON.parent, exist_ok=True)
    with open(ARCHITECTURES_JSON, 'w') as f:
        json.dump(architecture_samples, f)

    return architecture_samples


def conduct_rank_correlation(
        n: int,
        epochs: int,
        device: str = 'cuda'
):
    dataset = load_pamap2_dataset()
    training_loader, validation_loader, test_loader = dataset
    search_space = load_pamap2_search_space()

    supernet = Supernet(search_space=search_space).to(device)
    supernet_state_dict = torch.load(f=SUPERNET_PATH, map_location=device)
    supernet.load_state_dict(supernet_state_dict)
    sampled_architectures = get_or_create_architectures(supernet, n)

    if CHECKPOINT_CSV.exists():
        checkpoint_df = pd.read_csv(CHECKPOINT_CSV)

        if checkpoint_df.empty:
            completed_indices = set()
        else:
            completed_indices = checkpoint_df["index"].dropna().astype(int).tolist()

    else:
        completed_indices = set()

    for i, arch in enumerate(sampled_architectures):
        if i in completed_indices:
            continue

        print(f"Evaluating architecture {i + 1}/{len(sampled_architectures)}")
        subnet = Subnet(supernet=supernet, path=arch)
        supernet_summary = evaluate(
            model=subnet,
            evaluation_loader=validation_loader,
            device=device
        )
        supernet_acc = supernet_summary.accuracy

        standalone_model = construct_model(
            arch,
            in_dim=search_space["input"],
            out_dim=search_space["output"],
        ).to(device)

        trainer = ModelTrainer(
            model=standalone_model,
            dataset=dataset,
            epochs=epochs,
            device=device,
            logger=None,
            load_best_weights=True,
            evaluate_on_test_set=False
        )
        standalone_summary = trainer.run()
        standalone_acc = standalone_summary.accuracy

        new_row = pd.DataFrame([{
            "index": i,
            "supernet_acc": supernet_acc,
            "standalone_acc": standalone_acc
        }])
        new_row.to_csv(CHECKPOINT_CSV, mode="a", index=False, header=not CHECKPOINT_CSV.exists())
        print(f"Supernet: {supernet_acc:.2f} %, Standalone: {standalone_acc:.2f} %")

    acc_values_df = pd.read_csv(CHECKPOINT_CSV).head(n)
    supernet_acc_values = acc_values_df["supernet_acc"].tolist()
    standalone_acc_values = acc_values_df["standalone_acc"].tolist()

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
    for i, architecture_config in enumerate(sampled_architectures):
        subnet = Subnet(
            supernet=supernet,
            path=architecture_config,
        )

        summary = evaluate(
            model=subnet,
            evaluation_loader=validation_loader,
            device=device
        )

        print(f"Supernet Architecture Nr. {i + 1} of {len(sampled_architectures)}: {summary.accuracy:.2f} % Accuracy")
        supernet_acc_values.append(summary.accuracy)

    return supernet_acc_values


def evaluate_with_standalone_training(
        sampled_architectures: list,
        dataset: tuple[DataLoader, DataLoader, DataLoader],
        proxy_epochs: int,
        search_space: dict,
        device: str = 'cuda'
):
    standalone_values = []

    for i, architecture in enumerate(sampled_architectures):
        standalone_model = construct_model(
            architecture,
            in_dim=search_space['input'],
            out_dim=search_space['output']
        ).to(device)

        trainer = ModelTrainer(
            model=standalone_model,
            dataset=dataset,
            epochs=proxy_epochs,
            device=device,
            logger=None,
            load_best_weights=True,
            evaluate_on_test_set=False
        )

        summary = trainer.run()
        print(f"Standalone Architecture Nr. {i + 1} of {len(sampled_architectures)}: {summary.accuracy:.2f} % Accuracy")
        standalone_values.append(summary.accuracy)

    return standalone_values


if __name__ == "__main__":
    time_start = time.time()
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    rank_correlation = conduct_rank_correlation(n=500, epochs=50, device='cuda')
    rank_correlation.print()
    rank_correlation.write_to_csv()

    time_end = time.time()
    minutes, seconds = divmod(time_end - time_start, 60)
    print(f"Time: {int(minutes)}m {int(seconds)}s")
