"""
    Credit for this script is entirely reserved for natalie.maman@uni-due.de from the elastic-ai.explorer
    GitHub: https://github.com/es-ude/elastic-ai.explorer
"""

from torch import nn

from external.layer_adapter import (
    Conv2dToLSTMAdapter,
    LinearToConv2dAdapter,
    LinearToLstmAdapter,
    Conv2dToLinearAdapter,
    LSTMNoSequenceAdapter,
    LSTMToConv2dAdapter,
    ToLinearAdapter,
    Conv1dToLSTM
)

ADAPTER_REGISTRY = {
    ("conv2d", "lstm"): Conv2dToLSTMAdapter,
    ("linear", "conv2d"): LinearToConv2dAdapter,
    ("linear", "lstm"): LinearToLstmAdapter,
    ("conv2d", "linear"): Conv2dToLinearAdapter,
    ("lstm", "linear"): LSTMNoSequenceAdapter,
    ("lstm", "conv2d"): LSTMToConv2dAdapter,
    ("lstm", None): LSTMNoSequenceAdapter,
    (None, "linear"): ToLinearAdapter,
    ("*", "linear"): ToLinearAdapter,
    ("conv1d", "linear"): ToLinearAdapter,

    ("conv1d", "lstm"): Conv1dToLSTM,
    ("maxpool", "lstm"): Conv1dToLSTM,
    ("identity", "lstm"): Conv1dToLSTM,
}

activation_mapping = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanh": nn.Tanh(),
}

COMPOSITE_REGISTRY = {}
