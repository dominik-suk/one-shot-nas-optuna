from enum import Enum
from pathlib import Path

import optuna
import torch
from torch import nn

from external.build_model import construct_model
from external.sample_blocks import Sampler
from src.utils.yaml_io import load_pamap2_search_space
from src.paths import (
    BASELINE_BEST_MODEL_PATH,
    BASELINE_EXPERIMENT_DB_PATH,
    SPOS_BEST_MODEL_PATH,
    SPOS_EXPERIMENT_DB_PATH,
    SPOS_RANDOM_BEST_MODEL_PATH,
    SPOS_RANDOM_EXPERIMENT_DB_PATH,
)


class NAS_Algorithm(Enum):
    SPOS = "spos"
    BASELINE = "baseline"


class SearchStrategy(Enum):
    NSGAII = "nsgaii"
    RANDOM = "random"


def load_model(state_dict_path: Path, db_path: Path) -> nn.Module:
    study = optuna.load_study(storage=f"sqlite:///{db_path}")
    model = load_best_model(study=study)
    state_dict = torch.load(f=state_dict_path)
    model.load_state_dict(state_dict)
    return model


def load_best_model(study: optuna.Study) -> nn.Module:
    search_space = load_pamap2_search_space()
    best_trial = study.best_trial
    best_architecture = Sampler(best_trial).construct_sample(search_space=search_space)
    return construct_model(sample=best_architecture, in_dim=search_space["input"], out_dim=search_space["output"])


def get_study(path: Path | str) -> optuna.Study:
    return optuna.load_study(storage=f"sqlite:///{path}")


def load_spos_model(random_search: bool = False) -> nn.Module:
    if random_search:
        return load_model(state_dict_path=SPOS_RANDOM_BEST_MODEL_PATH, db_path=SPOS_RANDOM_EXPERIMENT_DB_PATH)
    return load_model(state_dict_path=SPOS_BEST_MODEL_PATH, db_path=SPOS_EXPERIMENT_DB_PATH)


def load_baseline_model() -> nn.Module:
    return load_model(state_dict_path=BASELINE_BEST_MODEL_PATH, db_path=BASELINE_EXPERIMENT_DB_PATH)
