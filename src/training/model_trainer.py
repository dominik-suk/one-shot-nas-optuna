import copy

from torch import nn
from torch.utils.data import DataLoader

from src.logging.summary import ModelSummary, HistoryTracker
from src.logging.train_logger import TrainLogger
from src.training.base_trainer import BaseTrainer
from src.training.evaluation import evaluate


class ModelTrainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            epochs: int,
            device: str = "cuda",
            logger: TrainLogger | None = None,
            load_best_weights: bool = True,
            evaluate_on_test_set: bool = False
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            epochs=epochs,
            device=device
        )

        self.logger = logger
        self.load_best_weights = load_best_weights
        self.evaluate_on_test = evaluate_on_test_set

        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_weights = copy.deepcopy(self.model.state_dict())
        self.tracker = HistoryTracker()

    def validate_epoch(self, epoch: int, train_summary: ModelSummary):
        summary = evaluate(self.model, self.validation_loader, self.criterion, self.device)

        if summary.accuracy > self.best_acc:
            self.best_acc = summary.accuracy
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.best_epoch = epoch

        if self.logger is not None:
            self.logger.log(
                epoch=epoch,
                total_epochs=self.epochs,
                train_acc=train_summary.accuracy,
                val_acc=summary.accuracy,
                best_acc=self.best_acc,
                best_epoch=self.best_epoch,
                val_loss=summary.loss,
                f1_score=summary.f1_score
            )

        self.tracker.update(epoch, train_summary, summary)

    def run(self) -> ModelSummary:
        for epoch in range(self.epochs):
            train_summary = self.train_one_epoch()
            self.validate_epoch(epoch, train_summary)

        if self.load_best_weights:
            self.model.load_state_dict(self.best_weights)

        if self.evaluate_on_test:
            evaluation_loader = self.test_loader
        else:
            evaluation_loader = self.validation_loader

        summary = evaluate(
            model=self.model,
            evaluation_loader=evaluation_loader,
            criterion=self.criterion,
            device=self.device
        )

        summary.history = self.tracker.to_dict()
        summary.print()
        return summary
