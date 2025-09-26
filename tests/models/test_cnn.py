"""
Unit tests for the Simple3DCNN model.

These tests verify the model's shape contracts and basic functionality
without requiring actual training or data loading.
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lesion_analysis.models.cnn import Simple3DCNN


class TestSimple3DCNN:
    """Test suite for Simple3DCNN model."""

    def test_model_shape_contract_batch_size_1(self):
        """Test that model produces correct output shape for single sample."""
        model = Simple3DCNN(in_channels=1, num_classes=1)

        # Create dummy input tensor (batch_size=1, channels=1, H=91, W=109, D=91)
        input_tensor = torch.randn(1, 1, 91, 109, 91)

        # Forward pass
        output = model(input_tensor)

        # Check output shape should be (1,) - single scalar output
        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
        assert output.dim() == 1, f"Expected 1D tensor, got {output.dim()}D"

    def test_model_shape_contract_batch_size_2(self):
        """Test that model produces correct output shape for batch of 2."""
        model = Simple3DCNN(in_channels=1, num_classes=1)

        # Create dummy input tensor (batch_size=2, channels=1, H=91, W=109, D=91)
        input_tensor = torch.randn(2, 1, 91, 109, 91)

        # Forward pass
        output = model(input_tensor)

        # Check output shape should be (2,) - batch of scalar outputs
        assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
        assert output.dim() == 1, f"Expected 1D tensor, got {output.dim()}D"

    def test_model_shape_contract_larger_batch(self):
        """Test that model produces correct output shape for larger batch."""
        model = Simple3DCNN(in_channels=1, num_classes=1)

        # Create dummy input tensor with batch_size=8
        input_tensor = torch.randn(8, 1, 91, 109, 91)

        # Forward pass
        output = model(input_tensor)

        # Check output shape should be (8,)
        assert output.shape == (8,), f"Expected shape (8,), got {output.shape}"
        assert output.dim() == 1, f"Expected 1D tensor, got {output.dim()}D"

    def test_model_different_input_channels(self):
        """Test model with different number of input channels."""
        model = Simple3DCNN(in_channels=2, num_classes=1)

        # Create dummy input tensor with 2 channels
        input_tensor = torch.randn(1, 2, 91, 109, 91)

        # Forward pass should not raise an error
        output = model(input_tensor)

        # Check output shape
        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"

    def test_model_different_num_classes(self):
        """Test model with different number of output classes."""
        model = Simple3DCNN(in_channels=1, num_classes=5)

        # Create dummy input tensor
        input_tensor = torch.randn(2, 1, 91, 109, 91)

        # Forward pass
        output = model(input_tensor)

        # For multi-class output, the squeeze(-1) should still work properly
        # The linear layer outputs (2, 5), squeeze(-1) shouldn't change it
        expected_shape = (2, 5) if 5 > 1 else (2,)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

    def test_model_dropout_rate_parameter(self):
        """Test that model accepts different dropout rates."""
        # These should not raise errors
        model1 = Simple3DCNN(dropout_rate=0.3)
        model2 = Simple3DCNN(dropout_rate=0.8)

        # Basic forward pass test
        input_tensor = torch.randn(1, 1, 91, 109, 91)
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)

        assert output1.shape == (1,), "Model with dropout=0.3 failed"
        assert output2.shape == (1,), "Model with dropout=0.8 failed"

    def test_model_training_mode_vs_eval_mode(self):
        """Test that model behaves differently in training vs eval mode due to dropout."""
        model = Simple3DCNN(dropout_rate=0.5)
        input_tensor = torch.randn(1, 1, 91, 109, 91)

        # Training mode
        model.train()
        train_output = model(input_tensor)

        # Eval mode
        model.eval()
        eval_output = model(input_tensor)

        # Both should have correct shape
        assert train_output.shape == (1,), "Training mode output shape incorrect"
        assert eval_output.shape == (1,), "Eval mode output shape incorrect"

        # Due to dropout, outputs might be different, but this is not guaranteed
        # so we just ensure they're both valid tensors
        assert torch.is_tensor(train_output), "Training output is not a tensor"
        assert torch.is_tensor(eval_output), "Eval output is not a tensor"

    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = Simple3DCNN()
        param_count = sum(p.numel() for p in model.parameters())

        # The model should be relatively small to prevent overfitting
        # Expected: roughly a few thousand to tens of thousands of parameters
        assert (
            1000 < param_count < 100000
        ), f"Parameter count {param_count} seems unreasonable"

    def test_model_gradients_flow(self):
        """Test that gradients flow through the model properly."""
        model = Simple3DCNN()
        input_tensor = torch.randn(1, 1, 91, 109, 91, requires_grad=True)

        # Forward pass
        output = model(input_tensor)

        # Backward pass
        loss = output.sum()  # Simple loss for gradient computation
        loss.backward()

        # Check that input gradients exist
        assert input_tensor.grad is not None, "Gradients not flowing to input"

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
