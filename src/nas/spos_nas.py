import optuna
from torch.utils.data import DataLoader

from src.logging.summary import ModelSummary
from src.models.supernet import Supernet
from src.training.evaluation import evaluate
from src.nas.nas_interface import NASExperiment
from src.training.supernet_trainer import SupernetTrainingWrapper


class SinglePathOneShotNASExperiment(NASExperiment):
    def __init__(
            self,
            supernet: Supernet,
            study: optuna.Study,
            search_space: dict,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            retraining_epochs: int,
            device: str = 'cuda'
    ):
        super().__init__(
            study=study,
            search_space=search_space,
            dataset=dataset,
            retraining_epochs=retraining_epochs,
            device=device
        )

        self.supernet = supernet

    def objective(self, trial: optuna.Trial):
        subnet_path = self.sample_architecture(trial)
        self.create_model(subnet_path)

        wrapped_supernet = SupernetTrainingWrapper(
            supernet=self.supernet,
            fixed_subnet_path=subnet_path,
        )

        summary: ModelSummary = evaluate(
            model=wrapped_supernet,
            evaluation_loader=self.validation_loader,
            criterion=self.criterion,
            device=self.device,
        )

        return summary.accuracy,
