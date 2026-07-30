"""Small CNN recognizer for the 5 selected Ranjana Lipi classes."""

from __future__ import annotations

import torch
from torch import nn


class RanjanaRecognizerCNN(nn.Module):
    """Lightweight CNN for 128x128 grayscale character recognition."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        # This custom CNN is intentionally small: the problem is only 5 classes
        # with a few thousand augmented samples, so a full ResNet would add
        # unnecessary capacity and overfitting risk for this phase.
        self.features = nn.Sequential(
            self._conv_block(1, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
