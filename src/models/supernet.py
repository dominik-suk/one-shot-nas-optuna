import torch
import torch.nn as nn

from src.models.choice_blocks import ChoiceBlock, DynamicConv1d, DynamicLSTM, DynamicLinear, GaussianDropout


class Supernet(nn.Module):
    def __init__(self, search_space: dict):
        super().__init__()
        self.search_space: dict = search_space
        self.in_sensors: int = self.search_space["input"][0]
        self.num_classes: int = search_space["output"]
        self.default_op_params: dict = search_space.get("default_op_params", {})
        self.max_channels: int = self._get_max_channels()
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
                            candidates["conv1d"] = DynamicConv1d(
                                max_channels=self.max_channels,
                                kernel_sizes=self.default_op_params["conv1d"]["kernel_size"]
                            )

                        case "lstm":
                            hidden_sizes = self.default_op_params["lstm"]["hidden_size"]
                            num_layers = self.default_op_params["lstm"]["num_layers"]
                            candidates["lstm"] = DynamicLSTM(
                                max_channels=self.max_channels,
                                max_hidden_size=max(hidden_sizes),
                                max_num_layers=max(num_layers)
                            )

                        case "linear":
                            candidates["linear"] = DynamicLinear(max_channels=self.max_channels)

                        case "maxpool":
                            candidates["maxpool"] = nn.MaxPool1d(2, 2)

                        case "dropout":
                            candidates["dropout"] = nn.Dropout(p=0.5)

                        case "gaussian_dropout":
                            candidates["gaussian_dropout"] = GaussianDropout(p=0.5)

                layer_list.append(ChoiceBlock(f"{block_id}_l{layer_index}", candidates))

            self.blocks[block_id] = layer_list

    def forward(self, x: torch.Tensor, model_sample: dict):
        # x: [Batch Size, Features, Sequence Length]
        for block_index, (block_id, layers) in enumerate(self.blocks.items()):
            if block_id not in model_sample:
                continue

            block_sample = model_sample[block_id]

            for layer_index, choice_block in enumerate(layers):
                if f"l{layer_index}" not in block_sample:
                    continue

                layer_sample = block_sample[f"l{layer_index}"]
                op_name = layer_sample["operation"]
                op_params = layer_sample["params"]

                if self._is_last_layer_of_last_block(block_index, layer_index, block_sample):
                    op_params["width"] = self.num_classes
                    op_params["activation"] = None

                x = choice_block(x, op_name, op_params)

        return x

    def _is_last_layer_of_last_block(self, block_index, layer_index, block_sample):
        return block_index >= len(self.blocks) - 1 and f"l{layer_index + 1}" not in block_sample

    @staticmethod
    def _is_last_layer(layer_index, block_sample):
        return f"l{layer_index + 1}" not in block_sample

    def _get_max_channels(self):
        channel_size_keys = {"out_channels", "hidden_size", "width"}
        max_channels = 0

        for op_name, params in self.default_op_params.items():
            for key, values in params.items():
                if key in channel_size_keys:
                    if isinstance(values, int):
                        max_channels = max(max_channels, values)
                    else:
                        max_channels = max(max_channels, max(values))

        if self._can_be_bidirectional():
            max_channels = max(max_channels, self._get_max_hidden_size() * 2)

        return max(max_channels, self.in_sensors)

    def _can_be_bidirectional(self) -> bool:
        possible_bidirectional_values: list[bool] | bool = self.default_op_params.get("lstm", {}).get("bidirectional", False)
        if isinstance(possible_bidirectional_values, list):
            return any(possible_bidirectional_values)
        return possible_bidirectional_values

    def _get_max_hidden_size(self) -> int:
        hidden_sizes = self.default_op_params.get("lstm", {}).get("hidden_size", 0)
        if isinstance(hidden_sizes, list):
            return max(hidden_sizes)
        return hidden_sizes
