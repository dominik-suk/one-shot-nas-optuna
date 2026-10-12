from collections import OrderedDict
from typing import Any

import optuna
from optuna.samplers import RandomSampler
from torch import nn

from external.build_model import construct_model, ShapeValueError
from external.sample_blocks import Sampler as SampleConstructor
from src.models.supernet import Supernet


class Subnet(nn.Module):
    def __init__(self, supernet: Supernet, path: dict):
        super().__init__()
        self.supernet = supernet
        self.path = path

    def forward(self, x):
        return self.supernet(x, self.path)


class SubnetPathGenerator:
    def __init__(self, search_space: dict, max_retries: int = 10):
        self.search_space = search_space
        self.max_retries = max_retries

        self.study = optuna.study.create_study(sampler=RandomSampler())

    def generate(self) -> OrderedDict[Any, Any]:
        for _ in range(self.max_retries):
            trial = self.study.ask()
            subnet_path = SampleConstructor(trial).construct_sample(self.search_space)

            if self.is_valid(subnet_path):
                self.study.tell(trial, values=0.0)
                return subnet_path

            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)

        raise RuntimeError(f"Couldn't generate a valid subnet path")

    def generate_from_fixed(self, fixed_architecture_config: dict):
        trial = self.study.ask()
        fixed_path = SampleConstructor(trial).construct_sample(fixed_architecture_config)
        self.study.tell(trial, values=0.0)

        return fixed_path

    def is_valid(self, sample: Any) -> bool:
        try:
            construct_model(sample, self.search_space["input"], self.search_space["output"])
            return True
        except ShapeValueError:
            return False