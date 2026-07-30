"""Convolutional autoencoder for per-character reconstruction modeling."""

from __future__ import annotations

import torch
from torch import nn


class RanjanaAutoencoder(nn.Module):
    """Lightweight 128x128 grayscale convolutional autoencoder.

    Phase 5 uses one independently trained instance per character. Keeping the
    model compact is intentional: each autoencoder only needs to learn the normal
    variation of one correct character form, not a broad visual vocabulary.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            self._encoder_block(1, 16),
            self._encoder_block(16, 32),
            self._encoder_block(32, 64),
            self._encoder_block(64, 128),
        )
        self.decoder = nn.Sequential(
            self._decoder_block(128, 64),
            self._decoder_block(64, 32),
            self._decoder_block(32, 16),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))
