"""
Custom loss functions for multi-task learning with conditional backpropagation.

This module provides loss functions that handle missing labels gracefully, enabling
training on datasets with partially available labels across different tasks.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def conditional_multitask_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    outcome_weight: float = 1.0,
) -> torch.Tensor:
    """
    Calculates a combined MSE loss for severity and outcome predictions.

    This function computes the loss only for samples with valid (non-NaN) labels,
    enabling training on the full dataset despite partial label availability.
    The outcome loss is only computed for samples that have valid outcome labels.

    Parameters
    ----------
    predictions : Dict[str, torch.Tensor]
        Dictionary containing 'severity' and 'outcome' prediction tensors
    targets : Dict[str, torch.Tensor]
        Dictionary containing 'severity', 'outcome', and 'treatment' target tensors.
        Outcome targets may contain NaN values for unlabeled samples.
    outcome_weight : float, default=1.0
        Weight multiplier for the outcome loss component

    Returns
    -------
    torch.Tensor
        Combined weighted loss scalar

    Examples
    --------
    >>> predictions = {"severity": torch.randn(4), "outcome": torch.randn(4)}
    >>> targets = {"severity": torch.randn(4), "outcome": torch.randn(4), "treatment": torch.zeros(4)}
    >>> loss = conditional_multitask_loss(predictions, targets)
    """
    # --- Severity Loss (computed for all samples in the batch) ---
    loss_severity = F.mse_loss(predictions["severity"], targets["severity"])

    # --- Outcome Loss (computed conditionally) ---
    outcome_preds = predictions["outcome"]
    outcome_targets = targets["outcome"]

    # Create a boolean mask for valid (non-NaN) outcome targets
    valid_mask = ~torch.isnan(outcome_targets)

    # Only compute loss if there are any valid targets in the batch
    if valid_mask.any():
        loss_outcome = F.mse_loss(
            outcome_preds[valid_mask], outcome_targets[valid_mask]
        )
    else:
        # If no valid targets in this batch, outcome loss is zero
        loss_outcome = torch.tensor(
            0.0, device=loss_severity.device, dtype=loss_severity.dtype
        )

    total_loss = loss_severity + outcome_weight * loss_outcome
    return total_loss
