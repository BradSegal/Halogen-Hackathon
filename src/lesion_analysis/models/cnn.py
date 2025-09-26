"""
Simple 3D CNN with aggressive regularization for lesion analysis.

This module implements a shallow 3D Convolutional Neural Network designed specifically
for small datasets with high dimensionality to prevent overfitting through extensive
regularization techniques.
"""

import torch
import torch.nn as nn
from typing import Dict


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
            nn.GroupNorm(
                num_groups=4, num_channels=8
            ),  # Batch-size independent normalization
            nn.ReLU(),
            nn.MaxPool3d(2),  # Aggressively downsample (91,109,91) -> (45,54,45)
            nn.Dropout3d(dropout_rate),  # High dropout rate
            # Block 2
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (22, 27, 22)
            nn.Dropout3d(dropout_rate),
            # Block 3
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=32),
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


class MultiTaskCNN(nn.Module):
    """
    Multi-task 3D CNN with shared backbone for simultaneous severity and outcome prediction.

    This architecture features a single shared convolutional backbone that learns rich
    spatial representations from all available lesion maps, feeding into two separate
    prediction heads:

    1. Severity Head: Predicts initial clinical_score based only on lesion anatomy
    2. Outcome Head: Predicts outcome_score conditioned on both anatomy AND treatment

    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels (typically 1 for binary lesion maps)
    backbone_features : int, default=32
        Number of features output by the shared backbone
    dropout_rate : float, default=0.5
        Dropout rate applied after each convolutional block

    Examples
    --------
    >>> model = MultiTaskCNN()
    >>> x = torch.randn(2, 1, 91, 109, 91)  # batch_size=2
    >>> w = torch.tensor([0, 1], dtype=torch.float32)  # treatments
    >>> output = model(x, w)  # dict with 'severity' and 'outcome' keys
    """

    def __init__(
        self,
        in_channels: int = 1,
        backbone_features: int = 32,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        # Shared Backbone extracts a feature vector from the 3D lesion map
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(16, backbone_features, kernel_size=3, padding=1),
            nn.GroupNorm(
                num_groups=min(16, backbone_features // 2),
                num_channels=backbone_features,
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_rate),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        # Head for predicting initial severity (prognostic)
        self.severity_head = nn.Linear(backbone_features, 1)

        # Head for predicting outcome, conditioned on anatomy AND treatment
        self.outcome_head = nn.Linear(
            backbone_features + 1, 1
        )  # +1 for the treatment variable

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task network.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape (batch_size, 1, 91, 109, 91)
        w : torch.Tensor
            Treatment tensor of shape (batch_size,)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with 'severity' and 'outcome' predictions, both of shape (batch_size,)
        """
        # x: image tensor, shape (batch, 1, 91, 109, 91)
        # w: treatment tensor, shape (batch,)
        features = self.backbone(x)

        # Severity prediction is unconditional on treatment
        pred_severity = self.severity_head(features).squeeze(-1)

        # For outcome, concatenate anatomical features with treatment variable
        outcome_head_input = torch.cat([features, w.unsqueeze(1)], dim=1)
        pred_outcome = self.outcome_head(outcome_head_input).squeeze(-1)

        return {"severity": pred_severity, "outcome": pred_outcome}
