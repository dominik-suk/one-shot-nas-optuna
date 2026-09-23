from torch import nn
from torch import optim


DEFAULT_OPTIMIZER_CONFIG = {
    "optimizer": optim.AdamW,
    "kwargs": {
        "lr": 5e-4,
        "weight_decay": 1e-3,
    }
}


DEFAULT_CRITERION_CONFIG = {
    "criterion": nn.CrossEntropyLoss,
    "kwargs": {
        "label_smoothing": 0.1,
    }
}


def get_default_optimizer(model: nn.Module) -> optim.Optimizer:
    optimizer = DEFAULT_OPTIMIZER_CONFIG["optimizer"]
    kwargs = DEFAULT_OPTIMIZER_CONFIG["kwargs"]

    return optimizer(model.parameters(), **kwargs)


def get_default_criterion() -> nn.Module:
    criterion = DEFAULT_CRITERION_CONFIG["criterion"]
    kwargs = DEFAULT_CRITERION_CONFIG["kwargs"]

    return criterion(**kwargs)
