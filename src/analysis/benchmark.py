import copy
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from tkinter import Tk
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from src.data.pamap2_labels import Pamap2ActivityType
from src.data.pamap2_loader import get_data
from src.logging.train_logger import TrainLogger
from src.models.train_model import train
from src.paths import BENCHMARK_SUMMARY_PATH, SUPERNET_TRAIN_TIME_LOGS
from src.utils.model_io import load_spos_model, load_baseline_model
from src.utils.optuna_io import get_spos_study, get_baseline_study


@dataclass
class BenchmarkSummary:
    timestamp: str
    gpu: str
    method: str
    search_strategy: str
    total_params: int
    flops: int
    supernet_training_minutes: float
    search_minutes: float
    latency_ms: float
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float

    def write_to_csv(self, path: Path = BENCHMARK_SUMMARY_PATH) -> None:
        df = pd.DataFrame([asdict(self)])
        file_exists = os.path.isfile(path)
        df.to_csv(path, mode='a', index=False, header=not file_exists)

    def print(self):
        print('\n'.join([
            f"Timestamp: {self.timestamp}",
            f"GPU: {self.gpu}",
            f"Method: {self.method}",
            f"Search Strategy: {self.search_strategy}",
            f"Total Params: {self.format_count(self.total_params)}",
            f"FLOPs: {self.format_count(self.flops)}",
            f"Supernet Training Minutes: {self.supernet_training_minutes:.2f}",
            f"Search Minutes: {self.search_minutes:.2f}",
            f"Latency Ms: {self.latency_ms:.3f}",
            f"Accuracy: {self.accuracy_mean:.2f} ± {self.accuracy_std:.2f} %",
            f"F1 Score: {self.f1_mean:.2f} ± {self.f1_std:.2f} %",
        ]), end='\n\n')


def format_count(count: int) -> str:
    prefixes = [
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    ]

    for factor, prefix in prefixes:
        if count >= factor:
            value = f"{count / factor:.2f}".rstrip("0").rstrip(".")

            return f"{value}{prefix}"

    return f"{count}"


def format_benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = pd.DataFrame()
    formatted["Method"] = df["method"]
    formatted["Search Strategy"] = df["search_strategy"]
    formatted["Params"] = df["total_params"].apply(lambda x: f"{format_count(x)}")
    formatted["FLOPs"] = df["flops"].apply(lambda x: f"{format_count(x)}")
    formatted["Latency"] = df["latency_ms"].apply(lambda x: f"{x:.3f} ms")
    formatted["Supernet Time"] = df["supernet_training_minutes"].apply(lambda x: f"{x:.1f} min" if x > 0 else "-")
    formatted["Search Time"] = df["search_minutes"].apply(lambda x: f"{x:.1f} min")
    formatted["Accuracy"] = df.apply(lambda row: f"{row['accuracy_mean']:.2f} ± {row['accuracy_std']:.2f} %", axis=1)
    formatted["Macro F1"] = df.apply(lambda row: f"{row['f1_mean']:.2f} ± {row['f1_std']:.2f} %", axis=1)

    return formatted


def copy_latex_table_to_clipboard(df: pd.DataFrame) -> None:
    r = Tk()
    r.withdraw()
    r.clipboard_clear()
    r.clipboard_append(df.to_latex())
    r.update()
    r.destroy()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        _ = model(input_tensor)
    return flop_counter.get_total_flops()


def measure_latency(model: nn.Module, input_tensor: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            _ = model(input_tensor)

        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / 100

        return latency_ms


def reset_model_weights(model: nn.Module) -> None:
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def retrain_and_evaluate(
        model: nn.Module,
        n_runs: int = 5,
        epochs: int = 30,
        seed: int = 42,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
        device: str = "cuda",
) -> tuple[tuple[float, float], tuple[float, float]]:
    accuracies = []
    f1_scores = []

    for _seed in range(seed, seed + n_runs):
        set_seed(_seed)
        run_model = copy.deepcopy(model)
        reset_model_weights(run_model)

        summary = train(
            model=run_model,
            epochs=epochs,
            activity_type=activity_type,
            load_best_weights=True,
            retraining_best_model=True,
            device=device,
            logger=TrainLogger(),
        )

        accuracies.append(summary.accuracy)
        f1_scores.append(summary.f1_score)

    mean_acc, std_acc = float(np.mean(accuracies)), float(np.std(accuracies))
    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))

    return (mean_acc, std_acc), (mean_f1, std_f1)


def calculate_search_cost(method: str, random_search: bool) -> float:
    if method == "SPOS":
        study = get_spos_study(random_search)
    else:
        study = get_baseline_study()

    start_time = min(trial.datetime_start for trial in study.trials)
    end_time = max(trial.datetime_complete for trial in study.trials)
    search_time_minutes = (end_time - start_time).total_seconds() / 60.0

    return search_time_minutes


def get_supernet_train_minutes(gpu: str) -> float:
    if not SUPERNET_TRAIN_TIME_LOGS.exists():
        return 0.0

    df = pd.read_csv(SUPERNET_TRAIN_TIME_LOGS)
    df = df[df["GPU"] == gpu]

    if df.empty:
        return 0.0

    return df["Training Minutes"].iloc[-1]


def benchmark(
        model: nn.Module,
        method: str,
        data_loader: DataLoader,
        device="cuda",
        random_search: bool = False,
        n_runs: int = 5,
        epochs: int = 50,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
) -> BenchmarkSummary:
    model = model.to(device)
    input_tensor, _ = next(iter(data_loader))
    input_tensor = input_tensor[:1].to(device)

    timestamp = datetime.now().strftime("%d.%m.%Y - %H:%M")
    gpu_name = torch.cuda.get_device_name(device)
    total_params, trainable_params = count_parameters(model)
    flops = count_flops(model, input_tensor)
    latency = measure_latency(model, input_tensor)

    (mean_acc, std_acc), (mean_f1, std_f1) = retrain_and_evaluate(
        model=model,
        n_runs=n_runs,
        epochs=epochs,
        seed=42,
        activity_type=activity_type,
        device=device,
    )

    search_minutes = calculate_search_cost(method, random_search)
    supernet_train_minutes = get_supernet_train_minutes(gpu_name) if method == "SPOS" else 0.0

    return BenchmarkSummary(
        timestamp=timestamp,
        gpu=gpu_name,
        method=method,
        search_strategy="Random" if random_search else "NSGA-II",
        total_params=total_params,
        flops=flops,
        latency_ms=latency,
        supernet_training_minutes=supernet_train_minutes,
        search_minutes=search_minutes,
        accuracy_mean=mean_acc,
        accuracy_std=std_acc,
        f1_mean=mean_f1,
        f1_std=std_f1
    )


def benchmark_spos(random_search: bool = False, do_save: bool = True, device: str = "cuda"):
    _, validation_loader, _ = get_data()
    model = load_spos_model(random_search=random_search)

    benchmark_summary = benchmark(
        model=model,
        method="SPOS",
        data_loader=validation_loader,
        device=device,
        random_search=random_search
    )
    benchmark_summary.print()

    if do_save:
        benchmark_summary.write_to_csv()


def benchmark_baseline(do_save: bool = True, device: str = "cuda"):
    _, validation_loader, _ = get_data()
    model = load_baseline_model()

    benchmark_summary = benchmark(
        model=model,
        method="Baseline",
        data_loader=validation_loader,
        device=device
    )
    benchmark_summary.print()

    if do_save:
        benchmark_summary.write_to_csv()


def main():
    benchmark_spos(random_search=False)
    benchmark_spos(random_search=True)
    benchmark_baseline()

    df = pd.read_csv(BENCHMARK_SUMMARY_PATH)
    df = format_benchmark_table(df)
    print(df.to_markdown())
    copy_latex_table_to_clipboard(df)


if __name__ == "__main__":
    main()
