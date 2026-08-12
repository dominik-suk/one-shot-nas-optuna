import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.choice_blocks import ChoiceBlock, DynamicConv1d, DynamicLinear, GaussianDropout


class Supernet(nn.Module):
    def __init__(self, search_space: dict):
        super().__init__()
        self.search_space: dict = search_space
        self.num_classes: int = search_space["output"]
        self.default_op_params: dict = search_space.get("default_op_params", {})
        input_shape = search_space["input"]
        in_sensors = input_shape[1]
        self.max_channels = self._get_max_channels()

        self.input = nn.Conv1d(in_sensors, self.max_channels, 1)
        self.blocks = nn.ModuleDict()

        for block_config in search_space.get("sequence", []):
            block_id = block_config["block"]
            op_candidates = block_config.get("op_candidates", [])
            if isinstance(op_candidates, str):
                op_candidates = [op_candidates]

            repeat_config = block_config.get("type_repeat", {})
            depth = repeat_config.get("depth", [1])
            max_depth = max(depth) if isinstance(depth, list) else depth

            layer_list = nn.ModuleList()

            for layer_index in range(max_depth):
                candidates = nn.ModuleDict()

                for op in op_candidates:
                    match op:
                        case "identity":
                            continue
                        case "conv1d":
                            kernel_sizes = self.default_op_params["conv1d"]["kernel_size"]
                            candidates["conv1d"] = DynamicConv1d(
                                in_channels_max=self.max_channels,
                                out_channels_max=self.max_channels,
                                kernel_sizes=kernel_sizes
                            )
                        case "maxpool":
                            candidates["maxpool"] = nn.MaxPool1d(1, 1)
                        case "lstm":
                            hidden_size_max = max(self.default_op_params["lstm"]["hidden_size"])
                            candidates["lstm"] = nn.LSTM(
                                input_size=self.max_channels,
                                hidden_size=hidden_size_max,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=True
                            )
                        case "dropout":
                            candidates["dropout"] = nn.Dropout(p=0.5)
                        case "gaussian_dropout":
                            candidates["gaussian_dropout"] = GaussianDropout(p=0.5)
                        case "linear":
                            candidates["linear"] = DynamicLinear(
                                self.max_channels,
                                self.max_channels
                            )
                layer_list.append(ChoiceBlock(f"{block_id}_l{layer_index}", candidates))
            self.blocks[block_id] = layer_list
        self.classifier = nn.Linear(self.max_channels, self.num_classes)

    def forward(self, x: torch.Tensor, model_sample: dict): # x: [Batch Size, Features, Sequence Length]
        x = self.input(x)

        for block_id, layers in self.blocks.items():
            if block_id not in model_sample:
                continue

            block_sample = model_sample[block_id]

            for layer_index, choice_block in enumerate(layers):
                if f"l{layer_index}" not in block_sample:
                    continue

                layer_sample = block_sample[f"l{layer_index}"]
                op_name = layer_sample["operation"]
                op_params = layer_sample["params"]

                match op_name:
                    case "lstm":
                        x = self._pad(x)
                        x = x.transpose(1, 2)
                        lstm_out, _ = choice_block.candidates["lstm"](x)
                        active_channel_count = lstm_out.shape[2] # todo: Channel size doesn't have to stay the same, can double
                        x = (lstm_out[:, :, :active_channel_count // 2] + lstm_out[:, :, active_channel_count // 2:]) / 2
                        x = x.transpose(1, 2)

                    case "linear":
                        x = self._flatten(x)
                        x = self._pad(x)
                        x = choice_block(x, op_name, op_params)

                    case _:
                        x = choice_block(x, op_name, op_params)

        x = self._flatten(x)
        x = self._pad(x)

        return self.classifier(x)

    @staticmethod
    def _flatten(x: torch.Tensor):
        if x.dim() == 3:
            x = x.mean(dim=-1)
        return x

    def _pad(self, x: torch.Tensor):
        current_channels = x.shape[1]
        if current_channels < self.max_channels:
            difference = self.max_channels - current_channels
            if x.dim() == 3:
                x = F.pad(x, (0, 0, 0, difference))
            else:
                x = F.pad(x, (0, difference))
        return x

    def _get_max_channels(self): # todo: max hidden size must be x2 if bidirectional=True
        max_channels = 0
        channel_size_keys = {"out_channels", "hidden_size", "width"}
        for op_name, params in self.default_op_params.items():
            for key, values in params.items():
                if key in channel_size_keys:
                    max_channels = max(max_channels, max(values))
        return max_channels