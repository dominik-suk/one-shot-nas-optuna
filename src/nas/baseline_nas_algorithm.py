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
            max_epochs: int = 50,
            n_proxy_epochs: int = 10
    ):
        super().__init__(study, search_space, activity_type, max_epochs)
        self.n_proxy_epochs = n_proxy_epochs

    def objective(self, trial: optuna.Trial) -> float:
        architecture = self.sample_architecture(trial)
        model = self.create_model(architecture)
        summary: ModelSummary = train(
            model=model,
            max_epochs=self.epochs,
            n_proxy_epochs=self.n_proxy_epochs,
            activity_type=self.activity_type,
            sequence_length=self.input_shape[1],
            load_best_weights=True,
            retraining_best_model=False,
            logger=TrainLogger(),
        )
        return summary.accuracy

    def get_best_architecture(self):
        return self.sample_architecture(self.study.best_trial)
