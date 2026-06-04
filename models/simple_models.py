"""W3: Simple PyTorch model definitions for MLIR export testing.

Each model is a self-contained nn.Module with an `example_input` property
that returns a representative input tensor for tracing/export.
"""
import torch
import torch.nn as nn


class LinearReLU(nn.Module):
    """Linear(2, 3) + ReLU — simplest MLIR export target."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))

    @property
    def example_input(self) -> torch.Tensor:
        return torch.randn(1, 2)


class ConvBNReLU(nn.Module):
    """Conv2d(1, 8, 3) + BatchNorm2d(8) + ReLU — realistic CNN building block."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

    @property
    def example_input(self) -> torch.Tensor:
        return torch.randn(1, 1, 16, 16)


class TwoLayerMLP(nn.Module):
    """Linear(4,8) + ReLU + Linear(8,2) — multi-layer perceptron."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

    @property
    def example_input(self) -> torch.Tensor:
        return torch.randn(1, 4)


# Registry of all test models
MODELS = {
    "linear_relu": LinearReLU,
    "conv_bn_relu": ConvBNReLU,
    "two_layer_mlp": TwoLayerMLP,
}
