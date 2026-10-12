import optuna
import torch
from torch.utils.data import DataLoader

from external.build_model import ShapeValueError
from src.analysis.benchmark import count_flops
from src.logging.summary import ModelSummary
from src.models.subnet import Subnet
from src.models.supernet import Supernet
from src.nas.nas_interface import NASExperiment
from src.training.evaluation import evaluate


class SinglePathOneShotNASExperiment(NASExperiment):
    def __init__(
            self,
            supernet: Supernet,
            study: optuna.Study,
            search_space: dict,
            dataset: tuple[DataLoader, DataLoader, DataLoader],
            retraining_epochs: int,
            max_flops: int | None = None,
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

        if max_flops is not None:
            self.max_flops = max_flops
        else:
            self.max_flops = float('inf')

    def objective(self, trial: optuna.Trial):
        subnet_path = self.sample_architecture(trial)

        if not self._path_is_valid(subnet_path):
            raise optuna.TrialPruned()

        subnet = Subnet(
            supernet=self.supernet,
            path=subnet_path
        )

        summary: ModelSummary = evaluate(
            model=subnet,
            evaluation_loader=self.validation_loader,
            device=self.device,
        )

        return summary.accuracy

    def _path_is_valid(self, subnet_path) -> bool:
        _input = torch.empty(1, *self.input_shape, device='cpu')

        try:
            model = self.create_model(subnet_path)
            flops = count_flops(model, _input)

            if flops < self.max_flops:
                print(flops)
                return True
            else:
                return False

        except ShapeValueError:
            return False
