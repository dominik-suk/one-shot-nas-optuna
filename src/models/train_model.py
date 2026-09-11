import copy
from typing import Sized

import optuna
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.pamap2_labels import Pamap2ActivityType
from src.data.pamap2_loader import get_data
from src.logging.train_logger import TrainLogger
from src.models.fixed_architecture import HumanActivityClassifier
from src.logging.summary import ModelSummary

def train_one_epoch(model: nn.Module, optimizer: Optimizer, criterion: nn.Module, training_loader: DataLoader, device: str):
    model.train()
    correct = 0

    for batch_index, (inputs, labels) in enumerate(training_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == labels).sum().item()

    assert isinstance(training_loader.dataset, Sized)
    accuracy = 100 * correct / len(training_loader.dataset)
    return accuracy


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cuda",
        criterion: nn.Module = nn.CrossEntropyLoss()
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
        max_epochs: int,
        device: str = "cuda",
        activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
        trial: optuna.Trial = None,
        proxy_epochs: int | None = None,
        logger: TrainLogger = None,
        sequence_length: int = 256
) -> ModelSummary:
    model.to(device)
    training_loader, validation_loader, test_loader = get_data(
        activity_type=activity_type,
        sequence_length=sequence_length
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_validation_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs = _get_epochs(max_epochs, proxy_epochs)

    for epoch in range(epochs):
        training_accuracy = train_one_epoch(model, optimizer, criterion, training_loader, device)
        val_summary = evaluate(model, validation_loader, device, criterion)

        if logger is not None:
            logger.on_epoch_end(
                epoch=epoch,
                total_epochs=epochs,
                train_acc=training_accuracy,
                val_acc=val_summary.accuracy,
                val_loss=val_summary.loss,
                f1_score=val_summary.f1_score,
            )

        if val_summary.loss < best_validation_loss:
            best_validation_loss = val_summary.loss
            best_weights = copy.deepcopy(model.state_dict())

        if trial is not None:
            trial.report(val_summary.loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        scheduler.step()

    model.load_state_dict(best_weights)
    summary = evaluate(
        model=model,
        criterion=criterion,
        data_loader=test_loader,
        device=device
    )

    if logger is not None:
        summary.print()

    return summary


def _get_epochs(max_epochs: int, proxy_epochs: int | None) -> int:
    if proxy_epochs is None:
        return max_epochs
    return proxy_epochs


def main():
    all_model = HumanActivityClassifier(num_classes=18)
    protocol_model = HumanActivityClassifier(num_classes=12)
    adl_model = HumanActivityClassifier(num_classes=6)
    train(all_model, max_epochs=50, device="cuda", activity_type=Pamap2ActivityType.ALL)
    train(protocol_model, max_epochs=50, device="cuda", activity_type=Pamap2ActivityType.PROTOCOL)
    train(adl_model, max_epochs=50, device="cuda", activity_type=Pamap2ActivityType.ADL)


if __name__ == "__main__":
    main()
