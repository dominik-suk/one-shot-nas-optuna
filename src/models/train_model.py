import copy
from typing import Sized

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.data.pamap2_labels import Pamap2ActivityType
from src.data.pamap2_loader import get_data
from src.logging.summary import ModelSummary
from src.logging.train_logger import TrainLogger
from src.models.fixed_architecture import HumanActivityClassifier


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
        epochs: int,
        activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
        n_proxy_epochs: int | None = None,
        sequence_length: int = 256,
        load_best_weights: bool = True,
        retraining_best_model: bool = False,
        device: str = "cuda",
        logger: TrainLogger = None,
) -> ModelSummary:
    model.to(device)
    training_loader, validation_loader, test_loader = get_data(
        activity_type=activity_type,
        sequence_length=sequence_length
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    summary = None
    best_validation_accuracy = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs = _get_epochs(epochs, n_proxy_epochs)

    for epoch in range(epochs):
        training_accuracy = train_one_epoch(model, optimizer, criterion, training_loader, device)
        summary = evaluate(model, validation_loader, device, criterion)

        if summary.accuracy > best_validation_accuracy:
            best_validation_accuracy = summary.accuracy
            best_weights = copy.deepcopy(model.state_dict())

        if logger is not None:
            logger.on_epoch_end(
                epoch=epoch,
                total_epochs=epochs,
                train_acc=training_accuracy,
                val_acc=summary.accuracy,
                current_best_acc=best_validation_accuracy,
                val_loss=summary.loss,
                f1_score=summary.f1_score,
            )

        if epoch >= 5:
            scheduler.step()

    if load_best_weights:
        model.load_state_dict(best_weights)

    if retraining_best_model:
        summary = evaluate(
            model=model,
            criterion=criterion,
            data_loader=test_loader,
            device=device
        )

    if logger is not None:
        summary.print()

    return summary


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
