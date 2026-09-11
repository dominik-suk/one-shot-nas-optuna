import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

import optuna
import torch
from torch import nn

from external.build_model import construct_model, ShapeValueError
from external.sample_blocks import Sampler
from src.data.pamap2_labels import Pamap2ActivityType
from src.logging.train_logger import TrainLogger
from src.models.train_model import train


class NASExperiment(ABC):
    def __init__(
            self,
            study: optuna.Study,
            search_space: dict,
            activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
            epochs: int = 50,
            device: str = 'cuda'
    ):
        self.study = study
        self.search_space = search_space
        self.input_shape = search_space["input"]
        self.output_shape = search_space["output"]
        self.activity_type = activity_type
        self.epochs = epochs
        self.device = device

    @abstractmethod
    def objective(self, trial: optuna.Trial):
        pass

    @abstractmethod
    def get_best_architecture(self):
        pass

    def run(self, n_trials: int = 100):
        current_trial_number = len(self.study.trials)
        remaining_trials = n_trials - current_trial_number
        if remaining_trials > 0:
            self.study.optimize(self.objective, n_trials=remaining_trials)

    def sample_architecture(self, trial):
        return Sampler(trial).construct_sample(self.search_space)

    def create_model(self, architecture_config: OrderedDict[Any, Any]) -> nn.Module:
        try:
            return construct_model(architecture_config, self.input_shape, self.output_shape)
        except ShapeValueError:
            raise optuna.TrialPruned()

    def get_best_model(self) -> nn.Module:
        return self.create_model(self.get_best_architecture())

    def train_best_model(self, save_path: str = None):
        best_model = self.get_best_model()
        train(best_model, max_epochs=self.epochs, activity_type=self.activity_type, logger=TrainLogger())
        if save_path:
            os.makedirs(Path(save_path).parent, exist_ok=True)
            torch.save(best_model.state_dict(), save_path)
