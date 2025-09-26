import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from src.lesion_analysis.models.cnn import MultiTaskCNN
from src.lesion_analysis.models.explanability import generate_saliency_map


class TestGenerateSaliencyMap:
    """Test suite for the generate_saliency_map function."""

    @pytest.fixture
    def model(self):
        """Create a MultiTaskCNN model for testing."""
        model = MultiTaskCNN()
        model.eval()
        return model

    @pytest.fixture
    def input_tensor(self):
        """Create a dummy input tensor with correct shape."""
        return torch.randn(1, 1, 91, 109, 91, requires_grad=True)

    @pytest.fixture
    def treatment_tensor(self):
        """Create a dummy treatment tensor."""
        return torch.tensor([1], dtype=torch.float32)

    @patch("src.lesion_analysis.models.explanability.IntegratedGradients")
    def test_generate_saliency_map_severity_head(
        self, mock_ig_class, model, input_tensor, treatment_tensor
    ):
        """Test saliency map generation for severity head."""
        # Setup mock
        mock_ig_instance = MagicMock()
        mock_ig_class.return_value = mock_ig_instance

        # Mock the attribute method to return a tensor of the correct shape
        mock_attributions = torch.randn(1, 1, 91, 109, 91)
        mock_delta = torch.tensor(0.001)
        mock_ig_instance.attribute.return_value = (mock_attributions, mock_delta)

        # Call the function
        saliency_map = generate_saliency_map(
            model=model,
            input_tensor=input_tensor,
            treatment_tensor=treatment_tensor,
            target_head="severity",
            n_steps=50,
        )

        # Assertions
        assert isinstance(saliency_map, np.ndarray)
        assert saliency_map.shape == (91, 109, 91)
        mock_ig_instance.attribute.assert_called_once()

    @patch("src.lesion_analysis.models.explanability.IntegratedGradients")
    def test_generate_saliency_map_outcome_head(
        self, mock_ig_class, model, input_tensor, treatment_tensor
    ):
        """Test saliency map generation for outcome head."""
        # Setup mock
        mock_ig_instance = MagicMock()
        mock_ig_class.return_value = mock_ig_instance

        # Mock the attribute method to return a tensor of the correct shape
        mock_attributions = torch.randn(1, 1, 91, 109, 91)
        mock_delta = torch.tensor(0.002)
        mock_ig_instance.attribute.return_value = (mock_attributions, mock_delta)

        # Call the function
        saliency_map = generate_saliency_map(
            model=model,
            input_tensor=input_tensor,
            treatment_tensor=treatment_tensor,
            target_head="outcome",
            n_steps=50,
        )

        # Assertions
        assert isinstance(saliency_map, np.ndarray)
        assert saliency_map.shape == (91, 109, 91)
        mock_ig_instance.attribute.assert_called_once()

    def test_invalid_target_head_raises_error(
        self, model, input_tensor, treatment_tensor
    ):
        """Test that invalid target_head raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            generate_saliency_map(
                model=model,
                input_tensor=input_tensor,
                treatment_tensor=treatment_tensor,
                target_head="invalid_head",
                n_steps=50,
            )

        assert "target_head must be either 'severity' or 'outcome'" in str(
            exc_info.value
        )

    @patch("src.lesion_analysis.models.explanability.IntegratedGradients")
    def test_model_forward_wrapper(
        self, mock_ig_class, model, input_tensor, treatment_tensor
    ):
        """Test that the model forward wrapper correctly passes tensors."""
        # Setup mock
        mock_ig_instance = MagicMock()
        mock_ig_class.return_value = mock_ig_instance

        # Store the forward wrapper when IntegratedGradients is initialized
        forward_wrapper = None

        def store_wrapper(fw):
            nonlocal forward_wrapper
            forward_wrapper = fw
            return mock_ig_instance

        mock_ig_class.side_effect = store_wrapper

        # Mock the attribute method
        mock_attributions = torch.randn(1, 1, 91, 109, 91)
        mock_delta = torch.tensor(0.001)
        mock_ig_instance.attribute.return_value = (mock_attributions, mock_delta)

        # Call the function
        generate_saliency_map(
            model=model,
            input_tensor=input_tensor,
            treatment_tensor=treatment_tensor,
            target_head="severity",
            n_steps=50,
        )

        # Verify the forward wrapper was created
        assert mock_ig_class.called
        assert callable(mock_ig_class.call_args[0][0])

    @patch("src.lesion_analysis.models.explanability.IntegratedGradients")
    def test_different_n_steps(
        self, mock_ig_class, model, input_tensor, treatment_tensor
    ):
        """Test that different n_steps values are passed correctly."""
        # Setup mock
        mock_ig_instance = MagicMock()
        mock_ig_class.return_value = mock_ig_instance

        # Mock the attribute method
        mock_attributions = torch.randn(1, 1, 91, 109, 91)
        mock_delta = torch.tensor(0.001)
        mock_ig_instance.attribute.return_value = (mock_attributions, mock_delta)

        # Test with different n_steps
        for n_steps in [10, 50, 100]:
            generate_saliency_map(
                model=model,
                input_tensor=input_tensor,
                treatment_tensor=treatment_tensor,
                target_head="severity",
                n_steps=n_steps,
            )

            # Check that n_steps was passed correctly
            call_kwargs = mock_ig_instance.attribute.call_args.kwargs
            assert call_kwargs["n_steps"] == n_steps

    @patch("src.lesion_analysis.models.explanability.IntegratedGradients")
    def test_baseline_is_zeros(
        self, mock_ig_class, model, input_tensor, treatment_tensor
    ):
        """Test that the baseline is an all-zeros tensor."""
        # Setup mock
        mock_ig_instance = MagicMock()
        mock_ig_class.return_value = mock_ig_instance

        # Mock the attribute method
        mock_attributions = torch.randn(1, 1, 91, 109, 91)
        mock_delta = torch.tensor(0.001)
        mock_ig_instance.attribute.return_value = (mock_attributions, mock_delta)

        # Call the function
        generate_saliency_map(
            model=model,
            input_tensor=input_tensor,
            treatment_tensor=treatment_tensor,
            target_head="severity",
            n_steps=50,
        )

        # Check that baseline is zeros
        call_kwargs = mock_ig_instance.attribute.call_args.kwargs
        baseline = call_kwargs["baselines"]
        assert torch.all(baseline == 0)
