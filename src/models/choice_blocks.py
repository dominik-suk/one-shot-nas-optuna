import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceBlock(nn.Module):
    def __init__(self, name: str, candidates: dict | nn.ModuleDict):
        super().__init__()
        self.name = name
        self.candidates = nn.ModuleDict(candidates)

    def forward(self, x: torch.Tensor, op_name: str, params: dict) -> torch.Tensor:
        if op_name == "identity" or op_name not in self.candidates:
            return x

        module = self.candidates[op_name]

        if isinstance(module, DynamicConv1d):
            out_channels = params["out_channels"]
            kernel_size = params["kernel_size"]
            stride = params["stride"]
            return module(x, active_out_channels=out_channels, active_kernel_size=kernel_size, stride=stride)

        elif isinstance(module, DynamicLinear):
            width = params["width"]
            return module(x, active_out_features=width)

        elif isinstance(module, nn.MaxPool1d):
            kernel_size = params["kernel_size"]
            stride = params["stride"]
            return F.max_pool1d(x, kernel_size=kernel_size, stride=stride)

        elif isinstance(module, (nn.Dropout, GaussianDropout)):
            p = params["p"]
            module.p = p
            return module(x)

        return module(x)


class DynamicConv1d(nn.Module):
    def __init__(self, in_channels_max, out_channels_max, kernel_sizes: list[int]):
        super().__init__()
        self.in_channels_max = in_channels_max
        self.out_channels_max = out_channels_max
        self.conv_dict = nn.ModuleDict()

        for kernel_size in kernel_sizes:
            self.conv_dict[str(kernel_size)] = nn.Conv1d(
                in_channels=in_channels_max,
                out_channels=out_channels_max,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )

    def forward(self, x: torch.Tensor, active_out_channels: int, active_kernel_size: int, stride: int = 1):
        active_in_channels = x.shape[1]
        conv = self.conv_dict[str(active_kernel_size)]
        weight = conv.weight[:active_out_channels, :active_in_channels, :]
        bias = conv.bias[:active_out_channels] if conv.bias is not None else None
        padding = active_kernel_size // 2

        return F.conv1d(x, weight, bias=bias, stride=stride, padding=padding)


class DynamicLinear(nn.Module):
    def __init__(self, in_features_max, out_features_max):
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features_max,
            out_features=out_features_max,
        )

    def forward(self, x: torch.Tensor, active_out_features: int):
        active_in_features = x.shape[1]
        weight = self.linear.weight[:active_out_features, :active_in_features]
        bias = self.linear.bias[:active_out_features] if self.linear.bias is not None else None

        return F.linear(x, weight, bias=bias)


class GaussianDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev + 1.0
            return x * epsilon
        else:
            return x
