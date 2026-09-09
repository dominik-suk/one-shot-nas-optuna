import os
from collections import OrderedDict
from typing import Any

import optuna
import torch
import torch.nn as nn
from optuna.samplers import RandomSampler

from external.sample_blocks import Sampler
from src.data.pamap2_labels import Pamap2ActivityType
from src.models.supernet import Supernet
from src.models.train_model import train
from src.logging.train_logger import TrainLogger


class SupernetTrainingWrapper(nn.Module):
    def __init__(self, supernet: Supernet, search_space: dict, fixed_architecture_config: OrderedDict[Any, Any] = None, ):
        super().__init__()
        self.supernet = supernet
        self.search_space = search_space
        self.fixed_architecture = self._init_fixed_architecture(fixed_architecture_config)

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
            architecture = self._take_one_sample()
        else:
            architecture = self.fixed_architecture

        return self.supernet(x, architecture)


def train_supernet(
        supernet: Supernet,
        search_space: dict,
        epochs: int = 100,
        device: str = "cuda",
        activity_type: Pamap2ActivityType = Pamap2ActivityType.ADL,
        fixed_architecture_config = None,
        save_dir: str = None,
):
    wrapped_supernet = SupernetTrainingWrapper(
        supernet,
        search_space,
        fixed_architecture_config=fixed_architecture_config
    )

    train(
        model=wrapped_supernet,
        max_epochs=epochs,
        device=device,
        activity_type=activity_type,
        logger=TrainLogger(),
        sequence_length=search_space["input"][1]
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'Supernet.pt')
        torch.save(supernet, save_path)
