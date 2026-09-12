import optuna

from src.data.pamap2_loader import get_data
from src.logging.summary import ModelSummary
from src.models.supernet import Supernet
from src.models.train_model import evaluate
from src.nas.nas_interface import NASExperiment
from src.nas.supernet_trainer import SupernetTrainingWrapper


class SinglePathOneShotNASExperiment(NASExperiment):
    def __init__(
            self,
            supernet: Supernet,
            study: optuna.Study,
            search_space: dict,
            epochs: int = 50,
            device: str = 'cuda'
    ):
        super().__init__(
            study=study,
            search_space=search_space,
            epochs=epochs,
            device=device
        )
        self.supernet = supernet
        self.data_loader = self._init_data_loader()

    def objective(self, trial: optuna.Trial):
        subnet_path = self.sample_architecture(trial)
        wrapped_supernet = SupernetTrainingWrapper(
            supernet=self.supernet,
            search_space=self.search_space,
            fixed_subnet_path=subnet_path,
        )
        summary: ModelSummary = evaluate(
            wrapped_supernet,
            data_loader=self.data_loader
        )
        summary.print()
        return summary.accuracy,

    def _init_data_loader(self):
        _, validation_loader, _ = get_data(
            activity_type=self.activity_type,
            sequence_length=self.sequence_length,
        )
        return validation_loader
