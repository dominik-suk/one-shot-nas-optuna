import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
from optuna.samplers import RandomSampler
from torch.utils.data import DataLoader

from external.build_model import construct_model, ShapeValueError
from external.sample_blocks import Sampler
from src.logging.summary import ModelSummary
from src.logging.supernet_logger import SupernetLogger
from src.models.supernet import Supernet
from src.training.base_trainer import BaseTrainer
from src.training.evaluation import evaluate
from src.utils.yaml_io import load_pamap2_fixed_arch_config


class SupernetTrainingWrapper(nn.Module):
    def __init__(
            self,
            supernet: Supernet,
            fixed_subnet_path: OrderedDict[Any, Any] | None = None,
    ):
        super().__init__()
        self.supernet = supernet
        self.search_space = supernet.search_space
        self.active_subnet_path = fixed_subnet_path

    def take_one_sample(self) -> OrderedDict[Any, Any]:
        while True:
            sample = self.get_new_sampler().construct_sample(self.search_space)
            if self.architecture_is_valid(sample):
                return sample

    def architecture_is_valid(self, sample):
        try:
            construct_model(sample, self.search_space["input"], self.search_space["output"])
            return True
        except ShapeValueError:
            return False

    @staticmethod
    def get_new_sampler():
        study = optuna.create_study(sampler=RandomSampler())
        trial = study.ask()
        return Sampler(trial)

    def forward(self, x):
        if self.training:
            return self.supernet(x, self.take_one_sample())
        else:
            return self.supernet(x, self.active_subnet_path)


class SupernetTrainer(BaseTrainer):
    def __init__(
            self,
            supernet: Supernet,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            epochs: int,
            device: str = "cuda",
            logger: SupernetLogger | None = None,
            n_val_architectures: int = 5
    ):
        super().__init__(
            model=SupernetTrainingWrapper(supernet),
            dataset=dataset,
            epochs=epochs,
            device=device,
        )

        self.logger = logger
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        fixed_architecture = self.model.get_new_sampler().construct_sample(load_pamap2_fixed_arch_config())
        self.val_architectures = [fixed_architecture] + [self.model.take_one_sample() for _ in range(n_val_architectures)]

    def evaluate_supernet(self) -> list[float]:
        self.model.eval()
        accuracies = []

        for arch in self.val_architectures:
            self.model.active_subnet_path = arch
            summary = evaluate(
                model=self.model,
                evaluation_loader=self.validation_loader,
                criterion=self.criterion,
                device=self.device,
            )
            accuracies.append(summary.accuracy)

        return accuracies

    def validate_epoch(self, epoch: int, train_summary: ModelSummary) -> None:
        accuracies = self.evaluate_supernet()

        if self.logger is not None:
            self.logger.log(
                epoch=epoch,
                total_epochs=self.epochs,
                train_acc=train_summary.accuracy,
                val_accuracies=accuracies
            )

    def save_weights(self, path: Path):
        os.makedirs(path.parent, exist_ok=True)
        torch.save(self.model.supernet.state_dict(), path)

    def run(self) -> None:
        for epoch in range(self.epochs):
            train_summary = self.train_one_epoch()
            self.validate_epoch(epoch, train_summary)
