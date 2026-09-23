import optuna
from torch.utils.data import DataLoader

from src.logging.train_logger import TrainLogger
from src.training.model_trainer import ModelTrainer
from src.nas.nas_interface import NASExperiment


class BaselineNASExperiment(NASExperiment):
    def __init__(
            self,
            study: optuna.Study,
            search_space: dict,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            n_proxy_epochs: int,
            retraining_epochs: int,
            device: str = "cuda",
    ):
        super().__init__(
            study=study,
            search_space=search_space,
            dataset=dataset,
            retraining_epochs=retraining_epochs,
            device=device
        )

        self.n_proxy_epochs = n_proxy_epochs

    def objective(self, trial: optuna.Trial) -> float:
        architecture = self.sample_architecture(trial)
        model = self.create_model(architecture)

        trainer = ModelTrainer(
            model=model,
            dataset=self.dataset,
            epochs=self.n_proxy_epochs,
            device=self.device,
            logger=TrainLogger(),
            load_best_weights=True,
            evaluate_on_test_set=False
        )

        summary = trainer.run()

        return summary.accuracy
