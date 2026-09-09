import optuna
import torch
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.pruners import HyperbandPruner
from torch import nn

from external.build_model import construct_model, ShapeValueError
from external.sample_blocks import Sampler
from src.data.pamap2_labels import Pamap2ActivityType
from src.logging.train_logger import TrainLogger
from src.models.train_model import train
from src.paths import PAMAP2_BASELINE_EXPERIMENT_DATABASE_PATH, BASELINE_MODEL_PATH
from src.utils.yaml_io import load_pamap2_search_space


class BaselineNASExperiment:
    def __init__(self, study: optuna.Study, search_space: dict, activity_type: Pamap2ActivityType = Pamap2ActivityType.ADL, epochs: int = 50, proxy: int = 10):
        self.study = study
        self.search_space = search_space
        self.input_shape = search_space["input"]
        self.output_shape = search_space["output"]
        self.activity_type = activity_type
        self.epochs = epochs
        self.proxy = proxy

    def run(self, n_trials: int = 100):
        current_trial_number = len(self.study.trials)
        remaining_trials = n_trials - current_trial_number
        if remaining_trials < 0:
            remaining_trials = 1
        self.study.optimize(self.objective, n_trials=remaining_trials)

    def objective(self, trial: optuna.Trial) -> float:
        model = self.create_model(trial)
        loss, _, _ = train(model, trial=trial, max_epochs=self.epochs, activity_type=self.activity_type, logger=TrainLogger())
        return loss

    def create_model(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> nn.Module:
        try:
            sampler = Sampler(trial)
            architecture_config = sampler.construct_sample(self.search_space)
            model = construct_model(architecture_config, self.input_shape, self.output_shape)
            return model
        except ShapeValueError:
            raise optuna.TrialPruned()

    def train_best_model(self, save_path: str):
        best_model = self.create_model(self.study.best_trial)
        train(best_model, max_epochs=self.epochs, activity_type=self.activity_type, logger=TrainLogger())
        torch.save(best_model.state_dict(), save_path)


def main():
    search_space = load_pamap2_search_space()
    db_url = f"sqlite:///{PAMAP2_BASELINE_EXPERIMENT_DATABASE_PATH}"
    optuna_sampler = TPESampler()
    # optuna_sampler = NSGAIISampler(population_size=20)
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
