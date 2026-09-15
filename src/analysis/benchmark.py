import os
from datetime import datetime
import pprint
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from src.data.pamap2_loader import get_data
from src.paths import BENCHMARK_SUMMARY_PATH
from src.utils.model_io import load_spos_model, load_baseline_model


@dataclass
class BenchmarkSummary:
    timestamp: str
    model_name: str | None
    gpu: str
    total_params: int
    trainable_params: int
    flops: int
    latency: float

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


def benchmark(model: nn.Module, model_name: str, data_loader: DataLoader, device="cuda") -> BenchmarkSummary:
    model = model.to(device)
    input_tensor, _ = next(iter(data_loader))
    input_tensor = input_tensor.to(device)

    timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    gpu_name = torch.cuda.get_device_name(device)
    total_params, trainable_params = count_parameters(model)
    flops = count_flops(model, input_tensor)
    latency = measure_latency(model, input_tensor)

    return BenchmarkSummary(
        timestamp=timestamp,
        model_name=model_name,
        gpu=gpu_name,
        total_params=total_params,
        trainable_params=trainable_params,
        flops=flops,
        latency=latency,
    )


def benchmark_spos(random_search: bool = False, do_save: bool = True, device: str = "cuda"):
    if random_search:
        model_name = "spos_random"

    else:
        model_name = "spos"

    _, validation_loader, _ = get_data()
    model = load_spos_model(random_search)

    benchmark_summary = benchmark(model, model_name, validation_loader, device=device)
    benchmark_summary.print()

    if do_save:
        benchmark_summary.write_to_csv()


def benchmark_baseline(do_save: bool = True, device: str = "cuda"):
    _, validation_loader, _ = get_data()
    model = load_baseline_model()

    benchmark_summary = benchmark(model, model_name="baseline", data_loader=validation_loader, device=device)
    benchmark_summary.print()

    if do_save:
        benchmark_summary.write_to_csv()


def main():
    benchmark_spos()
    benchmark_spos(random_search=True)
    benchmark_baseline()


if __name__ == "__main__":
    main()
