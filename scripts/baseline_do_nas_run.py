import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import NSGAIISampler

from src.nas.baseline_nas_algorithm import BaselineNASExperiment
from src.paths import PAMAP2_BASELINE_EXPERIMENT_DB_PATH, BASELINE_MODEL_PATH
from src.utils.yaml_io import load_pamap2_search_space


def main():
    search_space = load_pamap2_search_space()
    db_url = f"sqlite:///{PAMAP2_BASELINE_EXPERIMENT_DB_PATH}"
    optuna_sampler = NSGAIISampler(population_size=20)

    study = optuna.create_study(
        study_name='pamap2_baseline_nas',
        storage=db_url,
        load_if_exists=True,
        sampler=optuna_sampler,
        pruner=HyperbandPruner(
            min_resource=10,
            max_resource=50,
        ),
        direction='minimize'
    )
    experiment = BaselineNASExperiment(study, search_space)
    experiment.run(200)
    experiment.train_best_model(save_path=BASELINE_MODEL_PATH)


if __name__ == "__main__":
    main()