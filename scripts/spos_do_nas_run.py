import optuna
import torch
from optuna.samplers import NSGAIISampler

from src.data.pamap2_loader import load_pamap2_dataset as load_pamap2_dataset
from src.models.supernet import Supernet
from src.nas.spos_nas import SinglePathOneShotNASExperiment
from src.paths import SUPERNET_PATH, SPOS_EXPERIMENT_DB_PATH, SPOS_BEST_MODEL_PATH
from src.utils.yaml_io import load_pamap2_search_space


def main():
    search_space = load_pamap2_search_space()
    db_url = f"sqlite:///{SPOS_EXPERIMENT_DB_PATH}"

    dataset = load_pamap2_dataset(sequence_length=search_space['input'][1])

    supernet = Supernet(search_space)
    supernet.load_state_dict(torch.load(SUPERNET_PATH))

    study = optuna.create_study(
        study_name='pamap2_spos_nas',
        storage=db_url,
        load_if_exists=True,
        sampler=NSGAIISampler(population_size=50),
        direction='maximize'
    )

    experiment = SinglePathOneShotNASExperiment(
        supernet=supernet,
        study=study,
        search_space=search_space,
        dataset=dataset,
        retraining_epochs=50,
        device="cuda",
    )

    experiment.run(1000) # population_size * max_iterations => 50 * 20 = 1000
    experiment.train_best_model(save_path=SPOS_BEST_MODEL_PATH)


if __name__ == "__main__":
    main()