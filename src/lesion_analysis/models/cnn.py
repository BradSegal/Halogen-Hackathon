"""
Simple 3D CNN with aggressive regularization for lesion analysis.

This module implements a shallow 3D Convolutional Neural Network designed specifically
for small datasets with high dimensionality to prevent overfitting through extensive
regularization techniques.
"""

import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):
    """
    A simple, shallow 3D CNN designed to minimize overfitting.

    Architecture: 3 blocks of Conv-BatchNorm-ReLU-MaxPool-Dropout,
                  followed by a global average pooling and a single linear layer.

    This architecture prioritizes regularization over complexity to handle the challenge
    of learning from ~4k samples with ~900k input features (voxels).

    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels (typically 1 for binary lesion maps)
    num_classes : int, default=1
        Number of output classes/values (1 for both regression and binary classification)
    dropout_rate : float, default=0.5
        Dropout rate applied after each convolutional block

    Examples
    --------
    >>> model = Simple3DCNN()
    >>> x = torch.randn(2, 1, 91, 109, 91)  # batch_size=2
    >>> output = model(x)  # shape: (2,)
    """

    def __init__(
        self, in_channels: int = 1, num_classes: int = 1, dropout_rate: float = 0.5
    ):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),  # Normalizes activations, acts as regularizer
            nn.ReLU(),
            nn.MaxPool3d(2),  # Aggressively downsample (91,109,91) -> (45,54,45)
            nn.Dropout3d(dropout_rate),  # High dropout rate
            # Block 2
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (22, 27, 22)
            nn.Dropout3d(dropout_rate),
            # Block 3
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (11, 13, 11)
            nn.Dropout3d(dropout_rate),
            nn.AdaptiveAvgPool3d(1),  # Global Average Pooling to (32, 1, 1, 1)
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width, depth)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size,) for single output models
        """
        return self.net(x).squeeze(-1)  # Squeeze the final dimension
