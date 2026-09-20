import copy
from typing import Sized

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.pamap2_labels import Pamap2ActivityType
from src.data.pamap2_loader import get_data
from src.logging.summary import ModelSummary, HistoryTracker
from src.logging.supernet_logger import SupernetLogger
from src.logging.train_logger import TrainLogger
from src.models.fixed_architecture import HumanActivityClassifier


class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            epochs: int,
            data_loaders: tuple[DataLoader, DataLoader, DataLoader],
            device: str = "cuda",
            logger: TrainLogger | SupernetLogger | None = None,
            load_best_weights: bool = True,
    ):
        self.model = model.to(device)
        self.epochs = epochs
        self.train_loader, self.validation_loader, self.test_loader = data_loaders
        self.device = device
        self.logger = logger
        self.load_best_weights = load_best_weights

        self.epochs_without_improvement = 0
        self.patience_factor = 10
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-3)

        warmup_epochs = 5
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.05,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - warmup_epochs,
            eta_min=1e-6
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        self.best_acc = 0.0
        self.best_weights = copy.deepcopy(model.state_dict())
        self.tracker = HistoryTracker()

    def train_one_epoch(self) -> ModelSummary:
        self.model.train()
        correct = 0
        total_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if self.model.training:
                inputs = inputs + torch.randn_like(inputs) * 0.01

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

    def validate_epoch(self, epoch: int, train_summary: ModelSummary):
        summary = evaluate(self.model, self.validation_loader, self.criterion, self.device)

        if summary.accuracy > self.best_acc:
            self.best_acc = summary.accuracy
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.logger is not None:
            self.logger.log(
                epoch=epoch,
                total_epochs=self.epochs,
                train_acc=train_summary.accuracy,
                val_acc=summary.accuracy,
                current_best_acc=self.best_acc,
                val_loss=summary.loss,
                f1_score=summary.f1_score
            )

        self.tracker.update(epoch, train_summary, summary)

    def train_all_epochs(self):
        for epoch in range(self.epochs):
            train_summary = self.train_one_epoch()
            self.validate_epoch(epoch, train_summary)
            self.scheduler.step()

            if self.epochs_without_improvement >= self.patience_factor:
                break

    def run(self, evaluate_on_test: bool = False) -> ModelSummary:
        self.train_all_epochs()

        if self.load_best_weights:
            self.model.load_state_dict(self.best_weights)

        evaluation_loader = self.test_loader if evaluate_on_test else self.validation_loader
        summary = evaluate(self.model, evaluation_loader, self.criterion, self.device)
        summary.history = self.tracker.to_dict()
        summary.print()

        return summary


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1),
        device: str = "cuda"
) -> ModelSummary:
    model.eval()
    model.to(device)
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = sum(prediction == label for prediction, label in zip(all_predictions, all_labels))

    return ModelSummary(
        loss=total_loss / len(data_loader),
        accuracy=(correct / len(all_labels)) * 100.0,
        f1_score=f1_score(all_labels, all_predictions, average="macro") * 100.0
    )


def train(
        model: nn.Module,
        epochs: int,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
        n_proxy_epochs: int | None = None,
        sequence_length: int = 256,
        load_best_weights: bool = True,
        retraining_best_model: bool = False,
        data_loaders: tuple[DataLoader, DataLoader, DataLoader] = None,
        device: str = "cuda",
        logger: TrainLogger = None,
) -> ModelSummary:
    if data_loaders is None:
        data_loaders = get_data(activity_type=activity_type, sequence_length=sequence_length)

    epochs = _get_epochs(epochs, n_proxy_epochs)

    trainer = ModelTrainer(
        model=model,
        epochs=epochs,
        data_loaders=data_loaders,
        device=device,
        logger=logger,
        load_best_weights=load_best_weights
    )

    return trainer.run(evaluate_on_test=retraining_best_model)


def _get_epochs(max_epochs: int, n_proxy_epochs: int | None) -> int:
    if n_proxy_epochs is None:
        return max_epochs
    return n_proxy_epochs


def main():
    all_model = HumanActivityClassifier(num_classes=18)
    protocol_model = HumanActivityClassifier(num_classes=12)
    adl_model = HumanActivityClassifier(num_classes=6)
    train(all_model, epochs=50, device="cuda", activity_type=Pamap2ActivityType.ALL)
    train(protocol_model, epochs=50, device="cuda", activity_type=Pamap2ActivityType.PROTOCOL)
    train(adl_model, epochs=50, device="cuda", activity_type=Pamap2ActivityType.ADL)


if __name__ == "__main__":
    main()
