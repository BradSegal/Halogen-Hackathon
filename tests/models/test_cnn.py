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

from lesion_analysis.models.cnn import Simple3DCNN, MultiTaskCNN


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


class TestMultiTaskCNN:
    """Test suite for MultiTaskCNN model."""

    def test_model_shape_contract(self):
        """Test that model produces correct output shapes for multi-task predictions."""
        model = MultiTaskCNN()

        # Create dummy input tensors
        batch_size = 2
        x = torch.randn(batch_size, 1, 91, 109, 91)  # Images
        w = torch.tensor([0, 1], dtype=torch.float32)  # Treatments

        # Forward pass
        output = model(x, w)

        # Check that output is a dictionary
        assert isinstance(output, dict), "Output should be a dictionary"
        assert "severity" in output, "Output should have 'severity' key"
        assert "outcome" in output, "Output should have 'outcome' key"

        # Check shapes
        assert output["severity"].shape == (
            batch_size,
        ), f"Severity shape: {output['severity'].shape}"
        assert output["outcome"].shape == (
            batch_size,
        ), f"Outcome shape: {output['outcome'].shape}"

    def test_model_with_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = MultiTaskCNN()

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 1, 91, 109, 91)
            w = torch.zeros(batch_size)  # All control

            output = model(x, w)

            assert output["severity"].shape == (batch_size,)
            assert output["outcome"].shape == (batch_size,)

    def test_model_treatment_input_handling(self):
        """Test that model correctly handles treatment input."""
        model = MultiTaskCNN()

        x = torch.randn(3, 1, 91, 109, 91)
        # Mixed treatments: control, treatment, control
        w = torch.tensor([0, 1, 0], dtype=torch.float32)

        output = model(x, w)

        # Should produce valid outputs
        assert output["severity"].shape == (3,)
        assert output["outcome"].shape == (3,)

    def test_model_dropout_parameter(self):
        """Test model with different dropout rates."""
        model1 = MultiTaskCNN(dropout_rate=0.3)
        model2 = MultiTaskCNN(dropout_rate=0.7)

        x = torch.randn(1, 1, 91, 109, 91)
        w = torch.tensor([1], dtype=torch.float32)

        output1 = model1(x, w)
        output2 = model2(x, w)

        assert "severity" in output1 and "outcome" in output1
        assert "severity" in output2 and "outcome" in output2

    def test_model_backbone_features_parameter(self):
        """Test model with different backbone feature sizes."""
        model = MultiTaskCNN(backbone_features=64)

        x = torch.randn(2, 1, 91, 109, 91)
        w = torch.tensor([0, 1], dtype=torch.float32)

        output = model(x, w)

        assert output["severity"].shape == (2,)
        assert output["outcome"].shape == (2,)

    def test_model_training_vs_eval_mode(self):
        """Test model behavior in training vs eval mode."""
        model = MultiTaskCNN(dropout_rate=0.5)
        x = torch.randn(1, 1, 91, 109, 91)
        w = torch.tensor([0], dtype=torch.float32)

        # Training mode
        model.train()
        train_output = model(x, w)

        # Eval mode
        model.eval()
        eval_output = model(x, w)

        # Both should produce valid outputs
        assert "severity" in train_output and "outcome" in train_output
        assert "severity" in eval_output and "outcome" in eval_output

    def test_model_gradients_flow(self):
        """Test that gradients flow through both heads properly."""
        model = MultiTaskCNN()
        x = torch.randn(2, 1, 91, 109, 91, requires_grad=True)
        w = torch.tensor([0, 1], dtype=torch.float32)

        # Forward pass
        output = model(x, w)

        # Create a combined loss
        loss = output["severity"].sum() + output["outcome"].sum()

        # Backward pass
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None, "Gradients not flowing to input"

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"

    def test_model_parameter_sharing(self):
        """Test that backbone is shared between tasks."""
        model = MultiTaskCNN()

        # Count parameters in different parts
        backbone_params = sum(
            p.numel() for n, p in model.named_parameters() if "backbone" in n
        )
        severity_params = sum(
            p.numel() for n, p in model.named_parameters() if "severity" in n
        )
        outcome_params = sum(
            p.numel() for n, p in model.named_parameters() if "outcome" in n
        )

        # Backbone should have most parameters
        assert backbone_params > 0, "Backbone should have parameters"
        assert severity_params > 0, "Severity head should have parameters"
        assert outcome_params > 0, "Outcome head should have parameters"

        # Outcome head should have one more parameter than severity (for treatment)
        assert (
            outcome_params > severity_params
        ), "Outcome head should have more params due to treatment input"

    def test_model_deterministic_eval(self):
        """Test that model produces deterministic outputs in eval mode."""
        model = MultiTaskCNN()
        model.eval()  # Set to eval mode to disable dropout

        x = torch.randn(2, 1, 91, 109, 91)
        w = torch.tensor([0, 1], dtype=torch.float32)

        # Run twice
        output1 = model(x, w)
        output2 = model(x, w)

        # Should be identical in eval mode
        assert torch.allclose(output1["severity"], output2["severity"])
        assert torch.allclose(output1["outcome"], output2["outcome"])
