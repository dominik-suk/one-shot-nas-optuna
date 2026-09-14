from pathlib import Path

import optuna

from src.paths import BASELINE_EXPERIMENT_DB_PATH, SPOS_EXPERIMENT_DB_PATH, SPOS_RANDOM_EXPERIMENT_DB_PATH


def get_study(db_path: Path | str) -> optuna.Study:
    return optuna.load_study(study_name=None, storage=f"sqlite:///{db_path}")


def get_baseline_study() -> optuna.Study:
    return get_study(BASELINE_EXPERIMENT_DB_PATH)


def get_spos_study(random_search: bool = False) -> optuna.Study:
    if random_search:
        return get_study(SPOS_RANDOM_EXPERIMENT_DB_PATH)
    return get_study(SPOS_EXPERIMENT_DB_PATH)


def format_optuna_params(params: dict) -> dict:
    nested_params = {}
    for key, value in params.items():
        parts = key.split("/")
        current = nested_params

        for part in parts[:-1]:
            current = current.setdefault(part, {})

        current[parts[-1]] = value

    return nested_params
