import optuna

from src.data.pamap2_labels import Pamap2ActivityType
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
            activity_type: Pamap2ActivityType = Pamap2ActivityType.ADL,
            epochs: int = 50
    ):
        super().__init__(study, search_space, activity_type, epochs)
        self.supernet = supernet

    def objective(self, trial: optuna.Trial):
        supernet_path = self.sample_architecture(trial)
        wrapped_supernet = SupernetTrainingWrapper(
            self.supernet,
            self.search_space,
            supernet_path=supernet_path,
        )
        _, validation_loader, _ = get_data()
        summary: ModelSummary = evaluate(
            wrapped_supernet,
            data_loader=validation_loader
        )
        summary.print()
        return summary.loss,

    def get_best_architecture(self):
        return self.sample_architecture(self.study.best_trial)
