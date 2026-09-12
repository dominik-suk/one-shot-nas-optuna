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
from src.logging.train_logger import TrainLogger
from src.models.supernet import Supernet
from src.models.train_model import train
from src.paths import SUPERNET_PATH


class SupernetTrainingWrapper(nn.Module):
    def __init__(
            self,
            supernet: Supernet,
            search_space: dict,
            fixed_subnet_path: OrderedDict[Any, Any] | None = None,
            fixed_architecture_yaml: dict | None = None
    ):
        super().__init__()
        self.supernet = supernet
        self.search_space = search_space
        if fixed_subnet_path:
            self.fixed_subnet_path = fixed_subnet_path
        else:
            self.fixed_subnet_path = self._init_fixed_architecture(fixed_architecture_yaml)

    def _take_one_sample(self) -> OrderedDict[Any, Any]:
        return self._get_new_sampler().construct_sample(self.search_space)

    def _init_fixed_architecture(self, fixed_architecture_config):
        if fixed_architecture_config is not None:
            return self._get_new_sampler().construct_sample(fixed_architecture_config)
        return self._take_one_sample()

    @staticmethod
    def _get_new_sampler():
        study = optuna.create_study(sampler=RandomSampler())
        trial = study.ask()
        sampler = Sampler(trial)
        return sampler

    def forward(self, x):
        if self.training:
            return self.supernet(x, self._take_one_sample())
        else:
            return self.supernet(x, self.fixed_subnet_path)


def train_supernet(
        supernet: Supernet,
        search_space: dict,
        epochs: int = 100,
        device: str = "cuda",
        fixed_architecture_config: dict = None,
        save_path: str = SUPERNET_PATH,
        do_save: bool = True,
):
    wrapped_supernet = SupernetTrainingWrapper(
        supernet,
        search_space,
        fixed_architecture_yaml=fixed_architecture_config
    )

    train(
        model=wrapped_supernet,
        epochs=epochs,
        device=device,
        activity_type=get_activity_type_from_search_space(search_space),
        logger=TrainLogger(),
        sequence_length=search_space["input"][1],
        load_best_weights=False,
        retraining_best_model=False,
    )

    if do_save:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        torch.save(supernet.state_dict(), save_path)
