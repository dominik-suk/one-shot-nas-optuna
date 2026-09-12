import optuna

from src.logging.summary import ModelSummary
from src.logging.train_logger import TrainLogger
from src.models.train_model import train
from src.nas.nas_interface import NASExperiment


class BaselineNASExperiment(NASExperiment):
    def __init__(
            self,
            study: optuna.Study,
            search_space: dict,
            max_epochs: int = 50,
            n_proxy_epochs: int = 10,
            device: str = "cuda",
    ):
        super().__init__(
            study=study,
            search_space=search_space,
            epochs=max_epochs,
            device=device
        )
        self.n_proxy_epochs = n_proxy_epochs

    def objective(self, trial: optuna.Trial) -> float:
        architecture = self.sample_architecture(trial)
        model = self.create_model(architecture)
        summary: ModelSummary = train(
            model=model,
            epochs=self.epochs,
            n_proxy_epochs=self.n_proxy_epochs,
            activity_type=self.activity_type,
            sequence_length=self.sequence_length,
            load_best_weights=True,
            retraining_best_model=False,
            logger=TrainLogger(),
        )
        return summary.accuracy
