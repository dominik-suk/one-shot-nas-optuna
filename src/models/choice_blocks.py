import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceBlock(nn.Module):
    def __init__(self, name: str, candidates: dict | nn.ModuleDict):
        super().__init__()
        self.name = name
        self.candidates = nn.ModuleDict(candidates)

    def forward(self, x: torch.Tensor, op_name: str, params: dict) -> torch.Tensor:
        if op_name == "identity":
            return x
        elif op_name == "maxpool":
            return F.max_pool1d(x, kernel_size=params["kernel_size"], stride=params["stride"])
        elif op_name == "avgpool":
            return F.avg_pool1d(x, kernel_size=params["kernel_size"], stride=params["stride"])
        elif op_name == "dropout":
            return F.dropout(x, p=params["p"], training=self.training)
        elif op_name == "gaussian_dropout":
            return self.gaussian_dropout(x, params)

        if op_name in self.candidates:
            module = self.candidates[op_name]
            if isinstance(module, DynamicConv1d):
                return module(
                    x,
                    active_out_channels=params["out_channels"],
                    active_kernel_size=params["kernel_size"],
                    activation=params.get("activation", "relu"),
                    stride=params["stride"]
                )
            if isinstance(module, StaticLSTM):
                return module(
                    x,
                    active_hidden_size=params["hidden_size"],
                    num_layers=params["num_layers"],
                    bidirectional=params["bidirectional"]
                )
            if isinstance(module, DynamicLinear):
                return module(
                    x,
                    active_out_features=params["width"],
                    activation=params.get("activation", "relu")
                )

        raise ValueError(f"Unsupported operation: '{op_name}'")

    def gaussian_dropout(self, x, params: dict):
        if not self.training or params['p'] == 0:
            return x
        stddev = (params['p'] / (1.0 - params['p'])) ** 0.5
        return x * (1.0 + torch.randn_like(x) * stddev)


def apply_activation(x: torch.Tensor, activation: str):
    if activation == "relu":
        return torch.relu(x)
    elif activation == "tanh":
        return torch.tanh(x)
    return x


class DynamicConv1d(nn.Module):
    def __init__(self, max_channels: int, kernel_sizes: list[int]):
        super().__init__()
        self.conv_dict = nn.ModuleDict()

        for kernel_size in kernel_sizes:
            self.conv_dict[str(kernel_size)] = nn.Conv1d(
                in_channels=max_channels,
                out_channels=max_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

    def forward(self, x: torch.Tensor, active_out_channels: int, active_kernel_size: int,  activation: str | None, stride: int = 1):
        active_in_channels = x.shape[1]
        conv = self.conv_dict[str(active_kernel_size)]
        weight = conv.weight[:active_out_channels, :active_in_channels, :]
        bias = conv.bias[:active_out_channels] if conv.bias is not None else None
        padding = 0 if stride != 1 else active_kernel_size // 2

        x = F.conv1d(x, weight, bias=bias, stride=stride, padding=padding)
        x = apply_activation(x, activation)

        return x


class StaticLSTM(nn.Module):
    def __init__(self, max_channels: int, hidden_sizes: list[int], possible_num_layers: list[int]):
        super().__init__()
        self.max_channels = max_channels
        self.lstm_dict = nn.ModuleDict()

        for hidden_size in hidden_sizes:
            for layer in range(max(possible_num_layers)):
                in_channels = max_channels if layer == 0 else 2* hidden_size
                key = self._get_key(hidden_size, layer)

                self.lstm_dict[key] = nn.LSTM(
                    input_size=in_channels,
                    hidden_size=hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True
                )

    def forward(self, x: torch.Tensor, active_hidden_size: int, num_layers: int, bidirectional: bool):
        # x: [Batch, Channels, Length]
        input_size = x.shape[1]

        if input_size < self.max_channels:
            difference = self.max_channels - input_size
            x = F.pad(x, (0, 0, 0, difference))

        x = x.transpose(1, 2)
        # x: [Batch, Length, Channels]

        for layer in range(num_layers):
            key = self._get_key(active_hidden_size, layer)
            x, _ = self.lstm_dict[key](x)

        if not bidirectional:
            x = x[:, :, :active_hidden_size]

        x = x.transpose(1, 2)
        # x: [Batch, Channels, Length]

        return x

    @staticmethod
    def _get_key(h, l) -> str:
        return f"h{h}l{l}"


class DynamicLinear(nn.Module):
    def __init__(self, max_channels: int):
        super().__init__()
        self.linear = nn.Linear(
            in_features=max_channels,
            out_features=max_channels,
        )

    def forward(self, x: torch.Tensor, active_out_features: int, activation: str | None):
        x = self._flatten(x)
        active_in_features = x.shape[1]
        weight = self.linear.weight[:active_out_features, :active_in_features]
        bias = self.linear.bias[:active_out_features] if self.linear.bias is not None else None

        x = F.linear(x, weight, bias=bias)
        x = apply_activation(x, activation)

        return x

    @staticmethod
    def _flatten(x: torch.Tensor):
        # x: [Batch, Features, Sequence]
        if x.dim() == 3:
            x = x.mean(dim=2)
            # return x[:, :, -1]

        return x
