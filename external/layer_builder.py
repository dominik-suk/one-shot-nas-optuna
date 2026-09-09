"""
    Credit for this script is entirely reserved for natalie.maman@uni-due.de from the elastic-ai.explorer
    GitHub: https://github.com/es-ude/elastic-ai.explorer
"""

import copy
import math
from abc import abstractmethod, ABC
from typing import Sequence, Literal

from torch import nn as nn

from external.architecture_components import (
    SimpleLSTM,
    GaussianDropout,
    RepeatVector,
    TimeDistributed,
)
from external.registry import activation_mapping


LAYER_REGISTRY = {}

def _to_tuple(value, dims: int) -> tuple[int, ...]:
    """Ensure a value is a tuple of length `dims`."""
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,) * dims


def _conv_output_dim(in_dim, kernel, stride, padding, dilation):
    """Compute the output dimension for one spatial axis."""
    return (in_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def calculate_output_shape(
    shape: Sequence[int],
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int] | Literal["same"]= 0,
    dilation: int | Sequence[int] = 1,
    out_channels: int | None = None,
    layer_type: str = "conv2d",
):

    assert layer_type in {"conv1d", "conv2d", "pool1d", "pool2d"}, \
        f"Invalid layer type: {layer_type}"

    dims = 1 if "1d" in layer_type else 2


    kernel_size = _to_tuple(kernel_size, dims)
    stride = _to_tuple(stride, dims)

    dilation = _to_tuple(dilation, dims)
    new_shape = list(copy.deepcopy(shape))

    if "conv" in layer_type and out_channels is not None:
        new_shape[-(dims + 1)] = out_channels

    spatial_in = shape[-dims:]

    if padding == "same":
        new_spatial = [
            math.ceil(i / s)
            for i, s in zip(spatial_in, stride)
        ]

    else:
        padding = _to_tuple(padding, dims)
        new_spatial = [
            _conv_output_dim(i, k, s, p, d)
            for i, k, s, p, d in zip(spatial_in, kernel_size, stride, padding, dilation)
        ]

    new_shape[-dims:] = new_spatial
    return new_shape


def register_layer(name: str):
    """Decorator to register new layer types."""

    def wrapper(cls):
        LAYER_REGISTRY[name] = cls
        return cls

    return wrapper


class LayerBuilder(ABC):

    def build(self, input_shape, search_parameters: dict, output_shape=None):
        activation = search_parameters.get("activation", None)
        if output_shape is None:
            layer, shape = self.build_layer(input_shape, search_parameters)
        else:
            layer, shape = self.get_last_layer(
                input_shape, search_parameters, output_shape
            )

            if isinstance(shape, (list, tuple)) or shape != output_shape:
                in_features = math.prod(shape) if isinstance(shape, (list, tuple)) else shape

                layer = nn.Sequential(
                    layer,
                    nn.Flatten(),
                    nn.Linear(in_features, output_shape),
                )
                shape = output_shape

        if activation is not None:
            return nn.Sequential(layer, activation_mapping[activation]), shape
        return layer, shape

    @abstractmethod
    def build_layer(self, input_shape, search_parameters: dict):
        pass

    @abstractmethod
    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        pass


@register_layer("linear")
class LinearLayer(LayerBuilder):

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        linear = nn.Linear(input_shape, output_shape)
        return linear, output_shape

    def build_layer(self, input_shape, search_parameters: dict):
        linear = nn.Linear(input_shape, search_parameters["width"])
        return linear, search_parameters["width"]


class ConvLayer(LayerBuilder):

    conv_class: type[nn.Module] = None
    layer_type: str = None

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        return self.build_layer(input_shape, search_parameters)

    def validate_groups(
        self,
        in_channels,
        out_channels,
        groups,
    ):

        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by groups={groups}"
            )

        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by groups={groups}"
            )

    def get_out_channels_and_groups(self, input_shape, search_parameters: dict):
        groups = search_parameters.get("groups", 1)
        out_channels = search_parameters.get("out_channels", input_shape[0])

        if groups == "depthwise":
            groups = input_shape[0]
            out_channels = input_shape[0]
        self.validate_groups(input_shape[0], out_channels, groups)
        return out_channels, groups

    def get_padding(self, search_parameters: dict):
        padding = search_parameters.get("padding", 0)
        if search_parameters.get("stride", 1) != 1 and padding == "same":
            padding = 0
        return padding

    def build_layer(self, input_shape, search_parameters: dict):

        stride = search_parameters.get("stride", 1)

        out_channels, groups = self.get_out_channels_and_groups(
            input_shape, search_parameters
        )
        padding = self.get_padding(search_parameters)

        output_shape = calculate_output_shape(
            input_shape,
            search_parameters["kernel_size"],
            stride,
            padding=padding,
            out_channels=out_channels,
            layer_type=self.layer_type,
        )

        conv = self.conv_class(
            input_shape[0],
            out_channels,
            search_parameters["kernel_size"],
            stride,
            padding=padding,
            groups=groups,
        )

        return conv, output_shape


@register_layer("conv2d")
class Conv2dLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_class = nn.Conv2d
        self.layer_type = "conv2d"


@register_layer("conv1d")
class Conv1dLayer(ConvLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_class = nn.Conv1d
        self.layer_type = "conv1d"


@register_layer("lstm")
class LSTMLayer(LayerBuilder):
    def create_layer(
        self, input_shape, hidden_size, bidirectional, search_parameters: dict
    ):
        lstm = SimpleLSTM(
            input_shape[-1],
            hidden_size=hidden_size,
            num_layers=search_parameters.get("num_layers", 1),
            bidirectional=bidirectional,
            batch_first=search_parameters.get("batch_first", True),
            bias=search_parameters.get("bias", True),
            dropout=search_parameters.get("dropout", 0),
        )

        input_shape = [
            input_shape[0],
            hidden_size * 2 if bidirectional else hidden_size,
        ]
        return lstm, input_shape

    def build_layer(self, input_shape, search_parameters: dict):
        bidirectional: bool = search_parameters.get("bidirectional", False)
        hidden_size = search_parameters["hidden_size"]
        return self.create_layer(
            input_shape, hidden_size, bidirectional, search_parameters
        )

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        bidirectional: bool = search_parameters.get("bidirectional", False)
        hidden_size = output_shape
        if bidirectional:
            if (output_shape % 2) != 0:
                raise NotImplementedError
            else:
                hidden_size = output_shape // 2

        return self.create_layer(
            input_shape, hidden_size, bidirectional, search_parameters
        )


class PoolLayer(LayerBuilder):
    layer_map = {}
    param_keys = {"kernel_size": None, "stride": 1, "padding": 0}

    def build_layer(self, input_shape, search_parameters: dict):
        if isinstance(input_shape, int):
            return nn.Identity(), input_shape
        ndim = 2 if len(input_shape) == 3 else 1
        layer_cls = self.layer_map.get(f"{ndim}d", None)
        if layer_cls is None:
            raise ValueError(
                f"No matching class for {ndim}D in {self.__class__.__name__}"
            )

        pool = layer_cls(
            **{k: search_parameters.get(k, v) for k, v in self.param_keys.items()}
        )

        shape = calculate_output_shape(
            input_shape,
            search_parameters["kernel_size"],
            search_parameters.get("stride", self.param_keys["stride"]),
            search_parameters.get("padding", self.param_keys["padding"]),
            layer_type=f"conv{ndim}d",
        )
        return pool, shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        return self.build_layer(input_shape, search_parameters)


@register_layer("maxpool")
class MaxPoolLayer(PoolLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_map = {"1d": nn.MaxPool1d, "2d": nn.MaxPool2d}


@register_layer("avgpool")
class AvgPoolLayer(PoolLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_map = {"1d": nn.AvgPool1d, "2d": nn.AvgPool2d}


@register_layer("batch_norm")
class BatchNormLayer(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        num_features = (
            input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        )
        if isinstance(input_shape, int):
            layer_cls = nn.BatchNorm1d
        elif len(input_shape) == 3:
            layer_cls = nn.BatchNorm2d
        else:
            layer_cls = nn.BatchNorm1d

        return layer_cls(num_features=num_features), input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        self.build_layer(input_shape, search_parameters)


@register_layer("dropout")
class DropoutLayer(LayerBuilder):
    param_keys = ["p"]
    layer_map = {"1d": nn.Dropout, "2d": nn.Dropout2d}

    def build_layer(self, input_shape, search_parameters: dict):
        return nn.Dropout(search_parameters.get("p", 0.5)), input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        return self.build_layer(input_shape, search_parameters)


@register_layer("activation")
class ActivationLayer(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        return activation_mapping[search_parameters.get("op", "identity")], input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        return self.build_layer(input_shape, search_parameters)


@register_layer("layer_norm")
class LayerNorm(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        return nn.LayerNorm(input_shape), input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        self.build_layer(input_shape, search_parameters)


@register_layer("gaussian_dropout")
class GaussianDropoutLayer(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        return GaussianDropout(search_parameters.get("p", 0.5)), input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        return self.build_layer(input_shape, search_parameters)


@register_layer("repeat_vector")
class RepeatVectorLayer(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        times = search_parameters["times"]
        input_shape.append(times)
        return RepeatVector(times), input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        self.build_layer(input_shape, search_parameters)


@register_layer("time_distributed_linear")
class TimeDistributedLinear(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        batch_first = search_parameters.get("batch_first", True)

        output_sample_shape = [input_shape[0], search_parameters["width"]]

        module = nn.Sequential(
            nn.Linear(input_shape[-1], output_sample_shape[-1]),
            activation_mapping[search_parameters.get("activation", "identity")],
        )

        return TimeDistributed(module, batch_first=batch_first), output_sample_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        batch_first = search_parameters.get("batch_first", True)
        output_sample_shape = output_shape
        module = nn.Sequential(
            nn.Linear(input_shape[-1], output_sample_shape[-1]),
            activation_mapping[search_parameters.get("activation", "identity")],
        )

        return TimeDistributed(module, batch_first=batch_first), output_sample_shape


@register_layer("identity")
class IdentityLayer(LayerBuilder):
    def build_layer(self, input_shape, search_parameters: dict):
        return nn.Identity(), input_shape

    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        in_features = math.prod(input_shape) if isinstance(input_shape, (list, tuple)) else input_shape
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, output_shape),
        )
        return classifier, output_shape
