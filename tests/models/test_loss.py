"""
Unit tests for the conditional_multitask_loss function.

These tests verify that the loss function correctly handles missing labels
and computes the expected loss values.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lesion_analysis.models.loss import conditional_multitask_loss


class TestConditionalMultitaskLoss:
    """Test suite for conditional_multitask_loss function."""

    def test_loss_with_full_labels(self):
        """Test Case 1: All labels are present."""
        predictions = {
            "severity": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "outcome": torch.tensor([0.5, 1.5, 2.5, 3.5]),
        }

        targets = {
            "severity": torch.tensor([1.5, 2.5, 2.5, 3.5]),
            "outcome": torch.tensor([1.0, 1.0, 2.0, 3.0]),
            "treatment": torch.tensor([0, 1, 0, 1]),
        }

        loss = conditional_multitask_loss(predictions, targets, outcome_weight=1.0)

        # Calculate expected loss manually
        severity_loss = F.mse_loss(predictions["severity"], targets["severity"])
        outcome_loss = F.mse_loss(predictions["outcome"], targets["outcome"])
        expected_loss = severity_loss + outcome_loss

        assert torch.allclose(
            loss, expected_loss, rtol=1e-5
        ), f"Expected {expected_loss.item()}, got {loss.item()}"

    def test_loss_with_partial_labels(self):
        """Test Case 2: Some outcome targets are NaN."""
        predictions = {
            "severity": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "outcome": torch.tensor([0.5, 1.5, 2.5, 3.5]),
        }

        targets = {
            "severity": torch.tensor([1.5, 2.5, 2.5, 3.5]),
            "outcome": torch.tensor([1.0, np.nan, 2.0, np.nan]),  # Half are NaN
            "treatment": torch.tensor([0, 1, 0, 1]),
        }

        loss = conditional_multitask_loss(predictions, targets, outcome_weight=1.0)

        # Calculate expected loss manually
        severity_loss = F.mse_loss(predictions["severity"], targets["severity"])
        # Only compute outcome loss on valid (non-NaN) targets
        valid_mask = ~torch.isnan(targets["outcome"])
        outcome_loss = F.mse_loss(
            predictions["outcome"][valid_mask], targets["outcome"][valid_mask]
        )
        expected_loss = severity_loss + outcome_loss

        assert torch.allclose(
            loss, expected_loss, rtol=1e-5
        ), f"Expected {expected_loss.item()}, got {loss.item()}"

    def test_loss_with_no_outcome_labels(self):
        """Test Case 3: All outcome targets are NaN."""
        predictions = {
            "severity": torch.tensor([1.0, 2.0, 3.0]),
            "outcome": torch.tensor([0.5, 1.5, 2.5]),
        }

        targets = {
            "severity": torch.tensor([1.5, 2.5, 2.5]),
            "outcome": torch.tensor([np.nan, np.nan, np.nan]),  # All are NaN
            "treatment": torch.tensor([0, 1, 0]),
        }

        loss = conditional_multitask_loss(predictions, targets, outcome_weight=1.0)

        # When all outcome targets are NaN, outcome loss should be 0
        severity_loss = F.mse_loss(predictions["severity"], targets["severity"])
        expected_loss = severity_loss  # + 0 for outcome

        assert torch.allclose(
            loss, expected_loss, rtol=1e-5
        ), f"Expected {expected_loss.item()}, got {loss.item()}"

    def test_loss_with_different_outcome_weight(self):
        """Test that outcome_weight parameter works correctly."""
        predictions = {
            "severity": torch.tensor([1.0, 2.0]),
            "outcome": torch.tensor([0.5, 1.5]),
        }

        targets = {
            "severity": torch.tensor([2.0, 3.0]),
            "outcome": torch.tensor([1.5, 2.5]),
            "treatment": torch.tensor([0, 1]),
        }

        # Test with different weights
        for weight in [0.5, 1.0, 2.0]:
            loss = conditional_multitask_loss(
                predictions, targets, outcome_weight=weight
            )

            severity_loss = F.mse_loss(predictions["severity"], targets["severity"])
            outcome_loss = F.mse_loss(predictions["outcome"], targets["outcome"])
            expected_loss = severity_loss + weight * outcome_loss

            assert torch.allclose(
                loss, expected_loss, rtol=1e-5
            ), f"Weight {weight}: Expected {expected_loss.item()}, got {loss.item()}"

    def test_loss_with_single_sample(self):
        """Test loss computation with batch size of 1."""
        predictions = {
            "severity": torch.tensor([2.5]),
            "outcome": torch.tensor([1.5]),
        }

        targets = {
            "severity": torch.tensor([2.0]),
            "outcome": torch.tensor([1.0]),
            "treatment": torch.tensor([1]),
        }

        loss = conditional_multitask_loss(predictions, targets)

        # Should work without errors
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_loss_gradient_flow(self):
        """Test that gradients flow properly through the loss."""
        # Create tensors with requires_grad
        predictions = {
            "severity": torch.tensor([1.0, 2.0], requires_grad=True),
            "outcome": torch.tensor([0.5, 1.5], requires_grad=True),
        }

        targets = {
            "severity": torch.tensor([2.0, 3.0]),
            "outcome": torch.tensor([1.5, 2.5]),
            "treatment": torch.tensor([0, 1]),
        }

        loss = conditional_multitask_loss(predictions, targets)
        loss.backward()

        # Check gradients exist
        assert predictions["severity"].grad is not None, "Severity gradients missing"
        assert predictions["outcome"].grad is not None, "Outcome gradients missing"

    def test_loss_gradient_with_partial_labels(self):
        """Test gradient flow with NaN outcome labels."""
        predictions = {
            "severity": torch.tensor([1.0, 2.0], requires_grad=True),
            "outcome": torch.tensor([0.5, 1.5], requires_grad=True),
        }

        targets = {
            "severity": torch.tensor([2.0, 3.0]),
            "outcome": torch.tensor([1.5, np.nan]),  # Second is NaN
            "treatment": torch.tensor([0, 1]),
        }

        loss = conditional_multitask_loss(predictions, targets)
        loss.backward()

        # Gradients should exist
        assert predictions["severity"].grad is not None
        assert predictions["outcome"].grad is not None

        # For outcome, gradient should be zero for NaN target
        assert predictions["outcome"].grad[0] != 0, "First outcome should have gradient"
        assert (
            predictions["outcome"].grad[1] == 0
        ), "Second outcome gradient should be zero (NaN target)"

    def test_loss_device_compatibility(self):
        """Test that loss function works on different devices if available."""
        predictions = {
            "severity": torch.tensor([1.0, 2.0]),
            "outcome": torch.tensor([0.5, 1.5]),
        }

        targets = {
            "severity": torch.tensor([2.0, 3.0]),
            "outcome": torch.tensor([1.5, 2.5]),
            "treatment": torch.tensor([0, 1]),
        }

        # Test on CPU
        loss_cpu = conditional_multitask_loss(predictions, targets)
        assert loss_cpu.device.type == "cpu"

        # If CUDA is available, test on GPU
        if torch.cuda.is_available():
            predictions_gpu = {k: v.cuda() for k, v in predictions.items()}
            targets_gpu = {k: v.cuda() for k, v in targets.items()}
            loss_gpu = conditional_multitask_loss(predictions_gpu, targets_gpu)
            assert loss_gpu.device.type == "cuda"
            # Values should be the same
            assert torch.allclose(loss_cpu, loss_gpu.cpu())

    def test_loss_numerical_stability(self):
        """Test loss computation with extreme values."""
        # Test with very large values
        predictions = {
            "severity": torch.tensor([1e6, 2e6]),
            "outcome": torch.tensor([0.5e6, 1.5e6]),
        }

        targets = {
            "severity": torch.tensor([1.1e6, 2.1e6]),
            "outcome": torch.tensor([0.6e6, 1.6e6]),
            "treatment": torch.tensor([0, 1]),
        }

        loss = conditional_multitask_loss(predictions, targets)
        assert torch.isfinite(loss), "Loss should be finite even with large values"

        # Test with very small values
        predictions_small = {
            "severity": torch.tensor([1e-6, 2e-6]),
            "outcome": torch.tensor([0.5e-6, 1.5e-6]),
        }

        targets_small = {
            "severity": torch.tensor([1.1e-6, 2.1e-6]),
            "outcome": torch.tensor([0.6e-6, 1.6e-6]),
            "treatment": torch.tensor([0, 1]),
        }

        loss_small = conditional_multitask_loss(predictions_small, targets_small)
        assert torch.isfinite(
            loss_small
        ), "Loss should be finite even with small values"
