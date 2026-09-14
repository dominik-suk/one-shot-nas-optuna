from pathlib import Path

import optuna

from src.paths import BASELINE_EXPERIMENT_DB_PATH, SPOS_EXPERIMENT_DB_PATH, SPOS_RANDOM_EXPERIMENT_DB_PATH


def get_study(db_path: Path | str) -> optuna.Study:
    return optuna.load_study(storage=f"sqlite:///{db_path}")


def get_baseline_study() -> optuna.Study:
    return get_study(BASELINE_EXPERIMENT_DB_PATH)


def get_spos_study(random_search: bool) -> optuna.Study:
    if random_search:
        return get_study(SPOS_RANDOM_EXPERIMENT_DB_PATH)
    return get_study(SPOS_EXPERIMENT_DB_PATH)
