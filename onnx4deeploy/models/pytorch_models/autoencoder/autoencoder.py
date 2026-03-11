# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""FC Autoencoder for MLperf Tiny Anomaly Detection (AD) benchmark.

Reference: MLperf Tiny v1.0 Anomaly Detection task (ToyADMOS / DCASE 2020 Task 2).

The reference model is a fully-connected autoencoder trained on normal-class audio
features (128-dim log-mel spectrogram frames). Anomaly detection at inference time
is done by thresholding the reconstruction MSE loss.

Architecture (default — matches MLperf Tiny AD reference):
  Input  → 128 → 128 → 128 → 128 → 128 → Output  (symmetric encoder-decoder)
  Activations: ReLU on all hidden layers; linear output (no activation).
  All layers: nn.Linear, no BatchNorm (per reference model).

For PULP embedded deployment use the "tiny" variant (smaller hidden dims):
  Input → 64 → 32 → 16 → 32 → 64 → Output
"""

from typing import List

import torch
import torch.nn as nn


class FCAutoencoder(nn.Module):
    """
    Fully-connected symmetric autoencoder.

    Parameters
    ----------
    input_dim : int
        Feature vector length (128 for MLperf Tiny AD MFCC features).
    hidden_dims : list of int
        Encoder hidden dimensions (decoder mirrors these in reverse).
        E.g. [128, 128, 128] → encoder: input→128→128→128, decoder: 128→128→128→input.
    """

    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU(inplace=False))
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder, linear output)
        decoder_layers = []
        dims = list(reversed(hidden_dims)) + [input_dim]
        for i, out_dim in enumerate(dims):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 1:  # no activation on final output
                decoder_layers.append(nn.ReLU(inplace=False))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed feature vector; same shape as input."""
        z = self.encoder(x)
        return self.decoder(z)


def autoencoder_mlperf(input_dim: int = 128) -> FCAutoencoder:
    """
    MLperf Tiny AD reference autoencoder.

    5-layer FC autoencoder: input → [128, 128, 128] encoder → [128, 128, 128] decoder → output.
    ~100 K parameters at input_dim=128.
    """
    return FCAutoencoder(input_dim=input_dim, hidden_dims=[128, 128, 128])


def autoencoder_tiny(input_dim: int = 128) -> FCAutoencoder:
    """
    Tiny autoencoder for PULP embedded deployment.

    3-layer FC: input → [64, 32, 64] → output.  ~26 K parameters at input_dim=128.
    Fits comfortably in PULP L2 (< 100 KB).
    """
    return FCAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 64])
