import optuna
from optuna.samplers import NSGAIISampler

from src.data.pamap2_loader import load_pamap2_dataset
from src.nas.baseline_nas_algorithm import BaselineNASExperiment
from src.paths import BASELINE_EXPERIMENT_DB_PATH, BASELINE_BEST_MODEL_PATH
from src.utils.yaml_io import load_pamap2_search_space


def main():
    search_space = load_pamap2_search_space()
    dataset = load_pamap2_dataset()
    db_url = f"sqlite:///{BASELINE_EXPERIMENT_DB_PATH}"

    optuna_sampler = NSGAIISampler(
        population_size=10
    )

    study = optuna.create_study(
        study_name='pamap2_baseline_nas',
        storage=db_url,
        load_if_exists=True,
        sampler=optuna_sampler,
        direction='maximize'
    )

    experiment = BaselineNASExperiment(
        study=study,
        search_space=search_space,
        dataset=dataset,
        n_proxy_epochs=15,
        retraining_epochs=50,
        device="cuda",
    )

    experiment.run(n_trials=100)
    experiment.train_best_model(save_path=BASELINE_BEST_MODEL_PATH)


if __name__ == "__main__":
    main()