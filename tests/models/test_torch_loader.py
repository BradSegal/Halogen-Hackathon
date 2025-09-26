"""
Unit tests for the LesionDataset PyTorch Dataset class.

These tests verify that the dataset correctly loads data from files
and handles edge cases appropriately.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
import sys
import nibabel as nib

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lesion_analysis.models.torch_loader import LesionDataset


class TestLesionDataset:
    """Test suite for LesionDataset."""

    @pytest.fixture
    def mock_nifti_file(self):
        """Create a temporary mock NIfTI file for testing."""
        # Create dummy 3D data with the expected shape (91, 109, 91)
        dummy_data = np.random.randint(0, 2, size=(91, 109, 91)).astype(np.float32)

        # Create NIfTI image
        nifti_img = nib.Nifti1Image(dummy_data, affine=np.eye(4))

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_file:
            nib.save(nifti_img, temp_file.name)
            yield temp_file.name

        # Cleanup
        os.unlink(temp_file.name)

    @pytest.fixture
    def mock_dataframe(self, mock_nifti_file):
        """Create a mock DataFrame for testing."""
        return pd.DataFrame(
            {
                "lesion_id": ["lesion001.nii.gz", "lesion002.nii.gz"],
                "lesion_filepath": [mock_nifti_file, mock_nifti_file],
                "clinical_score": [5.0, 3.2],
                "outcome_score": [2.1, np.nan],  # Second has no outcome
                "is_responder": [True, False],
                "treatment_assignment": ["Treatment", "Control"],
            }
        )

    def test_dataset_initialization(self, mock_dataframe):
        """Test that dataset initializes correctly."""
        dataset = LesionDataset(mock_dataframe)

        assert len(dataset) == 2
        assert len(dataset.df) == 2

    def test_dataset_length(self, mock_dataframe):
        """Test that __len__ returns correct length."""
        dataset = LesionDataset(mock_dataframe)
        assert len(dataset) == 2

        # Test with single row
        single_row_df = mock_dataframe.iloc[:1].copy()
        single_dataset = LesionDataset(single_row_df)
        assert len(single_dataset) == 1

    def test_dataset_returns_dictionary_of_targets(self, mock_dataframe):
        """Test that __getitem__ returns a dictionary of targets."""
        dataset = LesionDataset(mock_dataframe)

        # Get first item
        data_tensor, targets = dataset[0]

        # Check data tensor shape (should have channel dimension)
        assert data_tensor.shape == (
            1,
            91,
            109,
            91,
        ), f"Expected shape (1, 91, 109, 91), got {data_tensor.shape}"
        assert data_tensor.dtype == torch.float32

        # Check that targets is a dictionary
        assert isinstance(targets, dict), "Targets should be a dictionary"
        assert "severity" in targets
        assert "outcome" in targets
        assert "treatment" in targets

        # Check severity target
        assert targets["severity"].shape == (), "Severity should be a scalar"
        assert targets["severity"].dtype == torch.float32
        assert pytest.approx(targets["severity"].item()) == 5.0  # First row's clinical_score

        # Check outcome target
        assert targets["outcome"].shape == (), "Outcome should be a scalar"
        assert targets["outcome"].dtype == torch.float32
        assert pytest.approx(targets["outcome"].item()) == 2.1  # First row's outcome_score

        # Check treatment target
        assert targets["treatment"].shape == (), "Treatment should be a scalar"
        assert targets["treatment"].dtype == torch.float32
        assert targets["treatment"].item() == 1.0  # "Treatment" -> 1

    def test_dataset_treatment_encoding(self, mock_dataframe):
        """Test that treatment assignment is correctly encoded."""
        dataset = LesionDataset(mock_dataframe)

        # Get first item (Treatment)
        _, targets1 = dataset[0]
        assert targets1["treatment"].item() == 1.0  # "Treatment" -> 1

        # Get second item (Control)
        _, targets2 = dataset[1]
        assert targets2["treatment"].item() == 0.0  # "Control" -> 0

    def test_dataset_handles_nan_targets(self, mock_nifti_file):
        """Test that dataset handles NaN targets correctly."""
        # Create DataFrame with NaN values
        df_with_nan = pd.DataFrame(
            {
                "lesion_id": ["lesion001.nii.gz"],
                "lesion_filepath": [mock_nifti_file],
                "clinical_score": [np.nan],
                "outcome_score": [np.nan],
                "treatment_assignment": [np.nan],
            }
        )

        dataset = LesionDataset(df_with_nan)
        data, targets = dataset[0]

        # Clinical score should be NaN when missing
        assert torch.isnan(targets["severity"]), "Missing clinical_score should be NaN"

        # Outcome score should be NaN when missing
        assert torch.isnan(targets["outcome"]), "Missing outcome_score should be NaN"

        # Treatment should default to 0 when missing
        assert (
            targets["treatment"].item() == 0.0
        ), "Missing treatment should default to 0"

    def test_dataset_data_range(self, mock_dataframe):
        """Test that loaded data has expected value range for binary lesion maps."""
        dataset = LesionDataset(mock_dataframe)
        data_tensor, _ = dataset[0]

        # Data should be binary (0s and 1s) since it's a lesion map
        unique_values = torch.unique(data_tensor)
        assert (
            len(unique_values) <= 2
        ), "Expected at most 2 unique values for binary lesion map"
        assert torch.all(unique_values >= 0), "All values should be >= 0"
        assert torch.all(unique_values <= 1), "All values should be <= 1"

    def test_dataset_index_out_of_bounds(self, mock_dataframe):
        """Test that dataset raises appropriate error for out-of-bounds index."""
        dataset = LesionDataset(mock_dataframe)

        with pytest.raises(IndexError):
            _ = dataset[5]  # Index too high

        with pytest.raises((IndexError, KeyError)):
            _ = dataset[-5]  # Negative index too low

    def test_dataset_with_empty_dataframe(self, mock_nifti_file):
        """Test dataset behavior with empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=[
                "lesion_id",
                "lesion_filepath",
                "clinical_score",
                "outcome_score",
                "treatment_assignment",
            ]
        )
        dataset = LesionDataset(empty_df)

        assert len(dataset) == 0

        with pytest.raises(IndexError):
            _ = dataset[0]

    def test_dataset_tensor_types(self, mock_dataframe):
        """Test that tensors have correct data types."""
        dataset = LesionDataset(mock_dataframe)
        data_tensor, targets = dataset[0]

        assert (
            data_tensor.dtype == torch.float32
        ), f"Expected float32, got {data_tensor.dtype}"

        # Check all target tensors
        for key in ["severity", "outcome", "treatment"]:
            assert (
                targets[key].dtype == torch.float32
            ), f"Expected float32 for {key}, got {targets[key].dtype}"

    def test_dataset_outcome_with_nan(self, mock_dataframe):
        """Test that dataset correctly handles NaN outcome for second sample."""
        dataset = LesionDataset(mock_dataframe)

        # Second sample should have NaN outcome
        _, targets = dataset[1]
        assert torch.isnan(targets["outcome"]), "Second sample should have NaN outcome"
        assert pytest.approx(targets["severity"].item()) == 3.2  # But severity should be valid

    def test_dataset_consistent_loading(self, mock_dataframe):
        """Test that loading the same item multiple times gives consistent results."""
        dataset = LesionDataset(mock_dataframe)

        # Load the same item multiple times
        data1, targets1 = dataset[0]
        data2, targets2 = dataset[0]

        # Should be identical
        assert torch.equal(data1, data2), "Data tensors should be identical"
        for key in targets1.keys():
            assert torch.equal(
                targets1[key], targets2[key]
            ), f"{key} tensors should be identical"

    def test_dataset_memory_efficiency(self, mock_dataframe):
        """Test that dataset doesn't preload all data (memory efficiency check)."""
        # This is more of a design verification test
        dataset = LesionDataset(mock_dataframe)

        # Dataset should only store the DataFrame
        # It should not have preloaded tensors stored as attributes
        assert hasattr(dataset, "df")

        # Should not have preloaded data attributes
        assert not hasattr(dataset, "_preloaded_data")
        assert not hasattr(dataset, "_cached_tensors")
