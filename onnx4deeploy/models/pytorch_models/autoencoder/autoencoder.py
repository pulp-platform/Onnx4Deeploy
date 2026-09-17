# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""FC Autoencoder for MLperf Tiny Anomaly Detection (AD) benchmark.

Reference: MLperf Tiny v1.0 Anomaly Detection task (ToyADMOS / DCASE 2020 Task 2).

The reference model is a fully-connected autoencoder trained on normal-class audio
features (640-dim vectors = 128 log-mel bins x 5 stacked frames). Anomaly detection at
inference time
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
import torch.nn.functional as F


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


class FrozenStatsBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that always normalizes with its running statistics.

    On-device training runs at batch 1, where per-batch statistics over a
    ``(N, C)`` tensor are undefined: the variance is identically zero and
    PyTorch refuses outright ("Expected more than 1 value per channel").
    The MLperf Tiny AD reference has BatchNorm after every hidden Dense, so
    rather than delete the layers we keep them -- learnable ``weight``/``bias``
    included, so the graph carries the reference's parameter count and the
    backward pass its gradients -- and normalize with the running statistics,
    which :func:`randomize_batchnorm_stats` seeds with random values (the
    fixtures are random-data fixtures throughout).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,  # never use batch statistics
            0.0,  # momentum unused
            self.eps,
        )


def randomize_batchnorm_stats(model: nn.Module, generator=None) -> nn.Module:
    """Seed every BatchNorm's running statistics with random values.

    ``running_mean`` ~ N(0, 0.1), ``running_var`` ~ U(0.5, 1.5) -- non-degenerate
    and strictly positive, so ``FrozenStatsBatchNorm1d`` normalizes with a
    well-conditioned scale instead of the default mean 0 / var 1 identity.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            with torch.no_grad():
                module.running_mean.normal_(0.0, 0.1, generator=generator)
                module.running_var.uniform_(0.5, 1.5, generator=generator)
    return model


class MLPerfADAutoencoder(nn.Module):
    """MLperf Tiny Anomaly Detection reference autoencoder (ToyADMOS / DCASE2020 Task 2).

    Faithful to ``mlcommons/tiny`` ``benchmark/training/anomaly_detection/keras_model.py``:

        Dense(h) -> BatchNorm -> ReLU   x n_enc      (h = 128)
        Dense(b) -> BatchNorm -> ReLU                (b = 8, bottleneck)
        Dense(h) -> BatchNorm -> ReLU   x n_enc
        Dense(input_dim)                             (linear reconstruction head)

    ``input_dim`` is 640 = 128 log-mel bins x 5 stacked frames, NOT a single
    128-bin frame. 10 Dense layers, ~268 K parameters.

    Parameters
    ----------
    input_dim : int
        Feature vector length (640 for the MLperf Tiny reference).
    hidden_dim : int
        Width of every non-bottleneck hidden layer (128).
    bottleneck_dim : int
        Width of the bottleneck layer (8).
    n_hidden_per_side : int
        Number of hidden layers before and after the bottleneck (4 each).
    use_batchnorm : bool
        Insert BatchNorm after every hidden Dense, as the reference does.
        The layers are :class:`FrozenStatsBatchNorm1d`, which normalizes with
        randomized running statistics so the model exports at batch 1.
        Set False only if the deployment target cannot lower 2-D BatchNorm.
    """

    def __init__(
        self,
        input_dim: int = 640,
        hidden_dim: int = 128,
        bottleneck_dim: int = 8,
        n_hidden_per_side: int = 4,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        def _hidden(in_dim: int, out_dim: int) -> List[nn.Module]:
            block: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if use_batchnorm:
                block.append(FrozenStatsBatchNorm1d(out_dim))
            block.append(nn.ReLU(inplace=False))
            return block

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_hidden_per_side):
            layers += _hidden(in_dim, hidden_dim)
            in_dim = hidden_dim
        layers += _hidden(in_dim, bottleneck_dim)  # bottleneck
        in_dim = bottleneck_dim
        for _ in range(n_hidden_per_side):
            layers += _hidden(in_dim, hidden_dim)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, input_dim))  # linear output

        self.net = nn.Sequential(*layers)
        if use_batchnorm:
            randomize_batchnorm_stats(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed feature vector; same shape as input."""
        return self.net(x)


def autoencoder_mlperf_ad(input_dim: int = 640, use_batchnorm: bool = True) -> MLPerfADAutoencoder:
    """MLperf Tiny AD reference autoencoder: 640 -> 128x4 -> 8 -> 128x4 -> 640.

    10 Dense layers, BatchNorm on every hidden layer, ~268 K parameters.
    This is the model the MLperf Tiny AUC 0.85 target is defined against.
    """
    return MLPerfADAutoencoder(input_dim=input_dim, use_batchnorm=use_batchnorm)


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
