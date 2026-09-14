import optuna
import torch
from optuna.samplers import RandomSampler

from src.models.supernet import Supernet
from src.nas.spos_nas import SinglePathOneShotNASExperiment
from src.paths import SUPERNET_PATH, SPOS_RANDOM_EXPERIMENT_DB_PATH, SPOS_RANDOM_BEST_MODEL_PATH
from src.utils.yaml_io import load_pamap2_search_space


def main():
    search_space = load_pamap2_search_space()
    db_url = f"sqlite:///{SPOS_RANDOM_EXPERIMENT_DB_PATH}"

    supernet = Supernet(search_space)
    supernet.load_state_dict(torch.load(SUPERNET_PATH))

    study = optuna.create_study(
        study_name='pamap2_random_spos_nas',
        storage=db_url,
        load_if_exists=True,
        sampler=RandomSampler(),
        direction='maximize'
    )

    experiment = SinglePathOneShotNASExperiment(
        supernet=supernet,
        study=study,
        search_space=search_space,
        device="cuda",
    )

    experiment.run(1000)
    experiment.train_best_model(save_path=SPOS_RANDOM_BEST_MODEL_PATH)


if __name__ == "__main__":
    main()