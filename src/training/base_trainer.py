from abc import ABC, abstractmethod
from typing import Sized

from torch.utils.data import DataLoader
from torch import nn

from src.logging.summary import ModelSummary
from src.training.hyperparameters import get_default_criterion, get_default_optimizer


class BaseTrainer(ABC):
    def __init__(
            self,
            model: nn.Module,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            epochs: int,
            device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.epochs = epochs
        self.device = device

        self.train_loader, self.validation_loader, self.test_loader = dataset
        self.criterion = get_default_criterion()
        self.optimizer = get_default_optimizer(self.model)

    def train_one_epoch(self) -> ModelSummary:
        self.model.train()
        correct = 0
        total_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        assert isinstance(self.train_loader.dataset, Sized)

        return ModelSummary(
            loss=total_loss / len(self.train_loader),
            accuracy=100 * correct / len(self.train_loader.dataset),
            f1_score=None
        )

    @abstractmethod
    def validate_epoch(self, epoch: int, train_summary: ModelSummary) -> ModelSummary:
        pass

    @abstractmethod
    def run(self):
        pass