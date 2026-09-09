import optuna
import torch
from optuna.samplers import NSGAIISampler

from src.models.supernet import Supernet
from src.nas.spos_nas import SinglePathOneShotNASExperiment
from src.paths import PAMAP2_SPOS_EXPERIMENT_DB_PATH, SUPERNET_PATH, SUPERNET_DIR
from src.utils.yaml_io import load_pamap2_search_space


def main():
    search_space = load_pamap2_search_space()
    db_url = f"sqlite:///{PAMAP2_SPOS_EXPERIMENT_DB_PATH}"
    optuna_sampler = NSGAIISampler(population_size=20)

    study = optuna.create_study(
        study_name='pamap2_spos_nas',
        storage=db_url,
        load_if_exists=True,
        sampler=optuna_sampler,
        direction='minimize'
    )
    supernet: Supernet = torch.load(SUPERNET_PATH, weights_only=False)
    experiment = SinglePathOneShotNASExperiment(
        supernet=supernet,
        study=study,
        search_space=search_space
    )
    experiment.run(200)
    experiment.train_best_model(save_path=SUPERNET_DIR / "BestArchitecture.pth")


if __name__ == "__main__":
    main()