import optuna

from src.data.pamap2_labels import Pamap2ActivityType
from src.logging.summary import ModelSummary
from src.logging.train_logger import TrainLogger
from src.models.train_model import train
from src.nas.nas_interface import NASExperiment


class BaselineNASExperiment(NASExperiment):
    def __init__(
            self,
            study: optuna.Study,
            search_space: dict,
            activity_type: Pamap2ActivityType = Pamap2ActivityType.PROTOCOL,
            epochs: int = 50,
            proxy: int = 10
    ):
        super().__init__(study, search_space, activity_type, epochs)
        self.proxy = proxy

    def objective(self, trial: optuna.Trial) -> float:
        architecture = self.sample_architecture(trial)
        model = self.create_model(architecture)
        summary: ModelSummary = train(
            model,
            trial=trial,
            max_epochs=self.epochs,
            activity_type=self.activity_type,
            logger=TrainLogger(),
            sequence_length=self.input_shape[1],
        )
        return summary.loss

    def get_best_architecture(self):
        return self.sample_architecture(self.study.best_trial)
