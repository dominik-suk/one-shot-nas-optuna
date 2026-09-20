import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
from optuna.samplers import RandomSampler
from torch.utils.data import DataLoader

from external.sample_blocks import Sampler
from src.data.pamap2_labels import get_activity_type_from_search_space
from src.data.pamap2_loader import get_data
from src.logging.summary import ModelSummary
from src.logging.supernet_logger import SupernetLogger
from src.models.supernet import Supernet
from src.models.train_model import evaluate, ModelTrainer
from src.paths import SUPERNET_PATH
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
        return self.get_new_sampler().construct_sample(self.search_space)

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


class SupernetTrainer(ModelTrainer):
    def __init__(
            self,
            supernet: SupernetTrainingWrapper,
            epochs: int,
            data_loaders: tuple[DataLoader, DataLoader, DataLoader],
            device: str = "cuda",
            logger: SupernetLogger = None,
            n_val_architectures: int = 5
    ):
        super().__init__(
            model=supernet,
            epochs=epochs,
            data_loaders=data_loaders,
            device=device,
            logger=logger,
            load_best_weights=False
        )

        assert isinstance(self.model, SupernetTrainingWrapper)
        fixed_architecture = self.model.get_new_sampler().construct_sample(load_pamap2_fixed_arch_config())
        self.val_architectures = [fixed_architecture] + [self.model.take_one_sample() for _ in range(n_val_architectures)]

    def evaluate_supernet(self) -> list[float]:
        assert isinstance(self.model, SupernetTrainingWrapper)

        self.model.eval()
        accuracies = []

        for arch in self.val_architectures:
            self.model.active_subnet_path = arch
            summary = evaluate(
                model=self.model,
                data_loader=self.validation_loader,
                criterion=self.criterion,
                device=self.device,
            )
            accuracies.append(summary.accuracy)

        return accuracies

    def validate_epoch(self, epoch: int, train_summary: ModelSummary):
        accuracies = self.evaluate_supernet()

        if self.logger is not None:
            self.logger.log(
                epoch=epoch,
                total_epochs=self.epochs,
                train_acc=train_summary.accuracy,
                val_accuracies=accuracies
            )

    def run(self, evaluate_on_test: bool = False) -> None:
        self.train_all_epochs()


def train_supernet(
        supernet: Supernet,
        epochs: int,
        device: str = "cuda",
        save_path: str = SUPERNET_PATH,
        do_save: bool = True,
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_space = supernet.search_space
    data_loaders = get_data(
        activity_type=get_activity_type_from_search_space(search_space),
        sequence_length=search_space["input"][1],
    )

    wrapped_supernet = SupernetTrainingWrapper(supernet)

    trainer = SupernetTrainer(
        supernet=wrapped_supernet,
        epochs=epochs,
        data_loaders=data_loaders,
        device=device,
        logger=SupernetLogger()
    )

    trainer.run()

    if do_save:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        torch.save(supernet.state_dict(), save_path)
