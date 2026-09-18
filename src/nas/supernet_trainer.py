import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
from optuna.samplers import RandomSampler

from external.sample_blocks import Sampler
from src.data.pamap2_labels import get_activity_type_from_search_space
from src.logging.supernet_logger import SupernetLogger
from src.models.supernet import Supernet
from src.models.train_model import train, evaluate
from src.paths import SUPERNET_PATH
from src.utils.yaml_io import load_pamap2_fixed_arch_config


class SupernetTrainingWrapper(nn.Module):
    def __init__(
            self,
            supernet: Supernet,
            fixed_subnet_path: OrderedDict[Any, Any] | None = None,
            n_val_architectures: int = 5
    ):
        super().__init__()
        self.supernet = supernet
        self.search_space = supernet.search_space

        if fixed_subnet_path is not None:
            self.val_architectures = [fixed_subnet_path]
        else:
            self.val_architectures = [self._get_new_sampler().construct_sample(load_pamap2_fixed_arch_config())] + [self.take_one_sample() for _ in range(n_val_architectures)]

        self.active_subnet_path = self.val_architectures[0]

    def take_one_sample(self) -> OrderedDict[Any, Any]:
        return self._get_new_sampler().construct_sample(self.search_space)

    @staticmethod
    def _get_new_sampler():
        study = optuna.create_study(sampler=RandomSampler())
        trial = study.ask()
        sampler = Sampler(trial)
        return sampler

    def forward(self, x):
        if self.training:
            return self.supernet(x, self.take_one_sample())
        else:
            return self.supernet(x, self.active_subnet_path)

    def evaluate_supernet(self, validation_loader, device="cuda"):
        self.eval()
        accuracies = []

        for arch in self.val_architectures:
            self.active_subnet_path = arch
            summary = evaluate(
                model=self,
                data_loader=validation_loader,
                device=device,
            )
            accuracies.append(summary.accuracy)

        return accuracies


def train_supernet(
        supernet: Supernet,
        epochs: int,
        device: str = "cuda",
        save_path: str = SUPERNET_PATH,
        do_save: bool = True,
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = supernet.search_space

    wrapped_supernet = SupernetTrainingWrapper(
        supernet,
        n_val_architectures=5,
    )

    train(
        model=wrapped_supernet,
        epochs=epochs,
        device=device,
        activity_type=get_activity_type_from_search_space(search_space),
        logger=SupernetLogger(),
        sequence_length=search_space["input"][1],
        load_best_weights=False,
        retraining_best_model=False,
    )

    if do_save:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        torch.save(supernet.state_dict(), save_path)
