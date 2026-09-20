import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from torch import nn

from src.logging.train_logger import TrainLogger
from src.models.train_model import train
from src.paths import TRAIN_HISTORY_DIR, GRAPHS_DIR
from src.utils.model_io import load_spos_model, load_best_model
from src.utils.plot_io import safe_save
from src.utils.optuna_io import get_spos_study, get_baseline_study


def plot_learning_curve(
        history: dict,
        model_name: str,
        do_save: bool = True,
        do_show: bool = True,
):
    epochs = history["epoch"]

    fig, (ax_loss, ax_acc) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Training Loss", color="blue", linewidth=1.8)
    ax_loss.plot(epochs, history["val_loss"], label="Validation Loss", color="purple", linewidth=1.8, linestyle="--")
    ax_loss.set_title(f"{model_name} Loss", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.grid(True, linestyle=":", alpha=0.5)
    ax_loss.legend(frameon=True, edgecolor="black")
    [spine.set_edgecolor("black") for spine in ax_loss.spines.values()]

    best_val_acc = max(history["val_acc"])
    best_epoch = epochs[history["val_acc"].index(best_val_acc)]

    ax_acc.plot(epochs, history["train_acc"], label="Training Accuracy", color="blue", linewidth=1.8)
    ax_acc.plot(epochs, history["val_acc"], label="Validation Accuracy", color="purple", linewidth=1.8, linestyle="--")
    ax_acc.axhline(y=best_val_acc, label=f"Best Validation Accuracy on Epoch {best_epoch}: {best_val_acc:.2f} %", color="red", linestyle=":", linewidth=1.0)
    ax_acc.set_title(f"{model_name} Accuracy", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.grid(True, linestyle=":", alpha=0.5)
    ax_acc.legend(frameon=True, edgecolor="black")
    [spine.set_edgecolor("black") for spine in ax_acc.spines.values()]

    plt.tight_layout()

    if do_save:
        safe_save(
            figure=fig,
            destination=GRAPHS_DIR / f"{model_name.replace(' ', '_')}_history.jpg"
        )

    if do_show:
        plt.show()

    plt.close(fig)

def save_to_csv(history: dict, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def load_from_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return df.to_dict(orient="list")


def get_history_from_model(
        model: nn.Module,
        model_name: str,
        epochs: int,
        device: str = "cuda",
        do_save: bool = True,
        load_existing: bool = True,
):
    csv_path = TRAIN_HISTORY_DIR / f"{model_name.replace(' ', '_')}_history.csv"

    if csv_path.exists() and load_existing:
        history = load_from_csv(csv_path)
        return history

    summary = train(
        model=model,
        epochs=epochs,
        device=device,
        retraining_best_model=True,
        logger=TrainLogger()
    )

    if do_save:
        save_to_csv(summary.history, csv_path)

    return summary.history


def plot_spos_learning_curve(random_search: bool = False):
    study = get_spos_study(random_search)
    model = load_best_model(study)
    model_name = "SPOS Random" if random_search else "SPOS NSGA-II"
    train_history = get_history_from_model(model=model, model_name=model_name, epochs=50, device="cuda", load_existing=False)
    plot_learning_curve(history=train_history, model_name=model_name, do_show=True, do_save=True)


def plot_baseline_learning_curve():
    study = get_baseline_study()
    model = load_best_model(study)
    model_name = "Baseline"
    train_history = get_history_from_model(model=model, model_name=model_name, epochs=50, device="cuda", load_existing=False)
    plot_learning_curve(history=train_history, model_name=model_name, do_show=True, do_save=True)
