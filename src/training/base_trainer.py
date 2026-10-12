from abc import ABC, abstractmethod

from torch import nn
from torch.utils.data import DataLoader

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

    @abstractmethod
    def train_epoch(self) -> ModelSummary | None:
        pass

    @abstractmethod
    def validate_epoch(self, epoch: int, train_summary: ModelSummary) -> ModelSummary | None:
        pass

    @abstractmethod
    def run(self):
        pass