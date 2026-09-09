import optuna
import torch
from torch import nn

from external.build_model import construct_model
from external.sample_blocks import Sampler
from src.data.pamap2_labels import Pamap2ActivityType
from src.logging.summary import ModelSummary
from src.logging.train_logger import TrainLogger
from src.models.train_model import train
from src.nas.nas_interface import NASExperiment


class SinglePathOneShotNASExperiment(NASExperiment):
    def __init__(
            self,
            study: optuna.Study,
            search_space: dict,
            activity_type: Pamap2ActivityType = Pamap2ActivityType.ADL,
            epochs: int = 50
    ):
        super().__init__(study, search_space, activity_type, epochs)

    def objective(self, trial: optuna.Trial):
        model = self.create_model(trial)
        summary: ModelSummary = train(
            model,
            trial=trial,
            max_epochs=self.epochs,
            activity_type=self.activity_type,
            logger=TrainLogger(),
            sequence_length=self.input_shape[1],
        )
        return summary.loss

    def create_model(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> nn.Module:
        sampler = Sampler(trial)
        architecture_config = sampler.construct_sample(self.search_space)
        model = construct_model(architecture_config, self.input_shape, self.output_shape)
        return model

    def train_best_model(self, save_path: str):
        best_model = self.create_model(self.study.best_trial)
        train(best_model, max_epochs=self.epochs, activity_type=self.activity_type, logger=TrainLogger())
        torch.save(best_model.state_dict(), save_path)