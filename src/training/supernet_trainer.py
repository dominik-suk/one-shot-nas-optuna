import os
from pathlib import Path
from typing import Sized

import optuna
import torch
from torch.utils.data import DataLoader

from src.logging.summary import ModelSummary
from src.logging.supernet_logger import SupernetLogger
from src.models.subnet import SubnetPathGenerator, Subnet
from src.models.supernet import Supernet
from src.training.base_trainer import BaseTrainer
from src.training.evaluation import evaluate
from src.utils.yaml_io import load_pamap2_fixed_arch_config


class SupernetTrainer(BaseTrainer):
    def __init__(
            self,
            supernet: Supernet,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            epochs: int,
            device: str = "cuda",
            logger: SupernetLogger | None = None,
            n_validation_architectures: int = 5
    ):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        super().__init__(
            model=supernet,
            dataset=dataset,
            epochs=epochs,
            device=device,
        )

        self.logger = logger
        self.n_validation_architectures = n_validation_architectures

        self.path_generator = SubnetPathGenerator(supernet.search_space)
        self.validation_architectures = self._init_validation_architectures()

    def train_epoch(self):
        self.model.train()
        correct = 0
        total_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            subnet = Subnet(
                supernet=self.model,
                path=self.path_generator.generate(),
            )

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = subnet(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        assert isinstance(self.train_loader.dataset, Sized)
        return ModelSummary(
            loss= total_loss / len(self.train_loader),
            accuracy= 100 * correct / len(self.train_loader.dataset),
            f1_score=None
        )

    def evaluate_supernet(self) -> list[float]:
        self.model.eval()
        accuracies = []

        for architecture in self.validation_architectures:
            subnet = Subnet(
                supernet=self.model,
                path=architecture
            )

            summary = evaluate(
                model=subnet,
                evaluation_loader=self.validation_loader,
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
        torch.save(self.model.state_dict(), path)

    def run(self) -> None:
        for epoch in range(self.epochs):
            train_summary = self.train_epoch()
            self.validate_epoch(epoch, train_summary)

    def _init_validation_architectures(self) -> list[dict]:
        fixed_architecture = load_pamap2_fixed_arch_config()

        return [
            self.path_generator.generate_from_fixed(fixed_architecture)
        ] + [
            self.path_generator.generate()
            for _ in range(self.n_validation_architectures)
        ]
