import copy
import os
from datetime import datetime
import pprint
from dataclasses import dataclass, asdict
from pathlib import Path

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
from src.paths import BENCHMARK_SUMMARY_PATH
from src.utils.model_io import load_spos_model, load_baseline_model


@dataclass
class BenchmarkSummary:
    timestamp: str
    gpu: str
    method: str
    search_strategy: str
    total_params: int
    flops: int
    latency_ms: float
    accuracy_mean: float
    accuracy_std: float

    def write_to_csv(self, path: Path = BENCHMARK_SUMMARY_PATH) -> None:
        df = pd.DataFrame([asdict(self)])
        file_exists = os.path.isfile(BENCHMARK_SUMMARY_PATH)
        df.to_csv(path, mode='a', index=False, header=not file_exists)

    def print(self):
        pprint.pprint(asdict(self))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def count_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    flop_counter = FlopCounterMode()
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


def retrain_and_evaluate(
        model: nn.Module,
        n_runs: int,
        epochs: int,
        seed: int,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
        device: str = "cuda",
) -> tuple[float, float]:
    accuracies = []

    for _seed in range(seed, seed + n_runs):
        torch.manual_seed(_seed)
        torch.cuda.manual_seed(_seed)
        np.random.seed(_seed)

        run_model = copy.deepcopy(model)
        reset_model_weights(run_model)

        summary = train(
            model=run_model,
            epochs=epochs,
            activity_type=activity_type,
            load_best_weights=True,
            device=device,
            logger=TrainLogger(),
        )
        accuracies.append(summary.accuracy)

    return float(np.mean(accuracies)), float(np.std(accuracies))


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
    input_tensor = input_tensor.to(device)

    timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    gpu_name = torch.cuda.get_device_name(device)
    total_params, trainable_params = count_parameters(model)
    flops = count_flops(model, input_tensor)
    latency = measure_latency(model, input_tensor)

    mean_acc, std_acc = retrain_and_evaluate(
        model=model,
        n_runs=n_runs,
        epochs=epochs,
        activity_type=activity_type,
        device=device,
    )

    return BenchmarkSummary(
        timestamp=timestamp,
        gpu=gpu_name,
        method=method,
        search_strategy="Random" if random_search else "NSGA-II",
        total_params=total_params,
        flops=flops,
        latency_ms=latency,
        accuracy_mean=mean_acc,
        accuracy_std=std_acc,
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


if __name__ == "__main__":
    main()
