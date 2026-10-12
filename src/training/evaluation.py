import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.logging.summary import ModelSummary
from src.training.hyperparameters import get_default_criterion


def evaluate(
        model: nn.Module,
        evaluation_loader: DataLoader,
        device: str = "cuda"
) -> ModelSummary:
    model.eval()
    model.to(device)
    criterion = get_default_criterion()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(evaluation_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = sum(prediction == label for prediction, label in zip(all_predictions, all_labels))

    return ModelSummary(
        loss=total_loss / len(evaluation_loader),
        accuracy=(correct / len(all_labels)) * 100.0,
        f1_score=f1_score(all_labels, all_predictions, average="macro") * 100.0
    )
