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
                "is_responder": [True, False],
                "treatment_assignment": ["Treatment", "Control"],
            }
        )

    def test_dataset_initialization(self, mock_dataframe):
        """Test that dataset initializes correctly."""
        dataset = LesionDataset(mock_dataframe, "clinical_score")

        assert len(dataset) == 2
        assert dataset.target_col == "clinical_score"
        assert len(dataset.df) == 2

    def test_dataset_length(self, mock_dataframe):
        """Test that __len__ returns correct length."""
        dataset = LesionDataset(mock_dataframe, "clinical_score")
        assert len(dataset) == 2

        # Test with single row
        single_row_df = mock_dataframe.iloc[:1].copy()
        single_dataset = LesionDataset(single_row_df, "clinical_score")
        assert len(single_dataset) == 1

    def test_dataset_getitem_clinical_score(self, mock_dataframe):
        """Test __getitem__ with clinical_score as target."""
        dataset = LesionDataset(mock_dataframe, "clinical_score")

        # Get first item
        data_tensor, target_tensor = dataset[0]

        # Check data tensor shape (should have channel dimension)
        assert data_tensor.shape == (
            1,
            91,
            109,
            91,
        ), f"Expected shape (1, 91, 109, 91), got {data_tensor.shape}"
        assert data_tensor.dtype == torch.float32

        # Check target tensor
        assert (
            target_tensor.shape == ()
        ), f"Expected scalar tensor, got shape {target_tensor.shape}"
        assert target_tensor.dtype == torch.float32
        assert target_tensor.item() == 5.0  # First row's clinical_score

    def test_dataset_getitem_is_responder(self, mock_dataframe):
        """Test __getitem__ with is_responder as target."""
        dataset = LesionDataset(mock_dataframe, "is_responder")

        # Get first item
        data_tensor, target_tensor = dataset[0]

        # Check data tensor shape
        assert data_tensor.shape == (1, 91, 109, 91)
        assert data_tensor.dtype == torch.float32

        # Check target tensor
        assert target_tensor.shape == ()
        assert target_tensor.dtype == torch.float32
        assert target_tensor.item() == 1.0  # True converted to 1.0

        # Get second item
        data_tensor_2, target_tensor_2 = dataset[1]
        assert target_tensor_2.item() == 0.0  # False converted to 0.0

    def test_dataset_handles_nan_targets_gracefully(self, mock_nifti_file):
        """Test that dataset handles NaN targets by converting them to a default value."""
        # Create DataFrame with NaN values
        df_with_nan = pd.DataFrame(
            {
                "lesion_id": ["lesion001.nii.gz"],
                "lesion_filepath": [mock_nifti_file],
                "clinical_score": [np.nan],
                "is_responder": [np.nan],
            }
        )

        # Test with NaN clinical_score - should convert to 0.0
        dataset_clinical = LesionDataset(df_with_nan, "clinical_score")
        data, target = dataset_clinical[0]  # Should NOT raise an error
        assert target.item() == 0.0, "NaN should be converted to 0.0"

        # Test with NaN is_responder - should convert to 0.0
        dataset_responder = LesionDataset(df_with_nan, "is_responder")
        data, target = dataset_responder[0]  # Should NOT raise an error
        assert target.item() == 0.0, "NaN should be converted to 0.0"

    def test_dataset_data_range(self, mock_dataframe):
        """Test that loaded data has expected value range for binary lesion maps."""
        dataset = LesionDataset(mock_dataframe, "clinical_score")
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
        dataset = LesionDataset(mock_dataframe, "clinical_score")

        with pytest.raises(IndexError):
            _ = dataset[5]  # Index too high

        with pytest.raises((IndexError, KeyError)):
            _ = dataset[-5]  # Negative index too low

    def test_dataset_with_empty_dataframe(self, mock_nifti_file):
        """Test dataset behavior with empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=["lesion_id", "lesion_filepath", "clinical_score"]
        )
        dataset = LesionDataset(empty_df, "clinical_score")

        assert len(dataset) == 0

        with pytest.raises(IndexError):
            _ = dataset[0]

    def test_dataset_tensor_types(self, mock_dataframe):
        """Test that tensors have correct data types."""
        dataset = LesionDataset(mock_dataframe, "clinical_score")
        data_tensor, target_tensor = dataset[0]

        assert (
            data_tensor.dtype == torch.float32
        ), f"Expected float32, got {data_tensor.dtype}"
        assert (
            target_tensor.dtype == torch.float32
        ), f"Expected float32, got {target_tensor.dtype}"

    def test_dataset_with_different_target_columns(self, mock_dataframe):
        """Test dataset with different target column names."""
        # Add a custom target column
        mock_dataframe["custom_target"] = [10.5, 20.3]

        dataset = LesionDataset(mock_dataframe, "custom_target")
        _, target = dataset[0]

        assert target.item() == 10.5, "Should use the specified target column"

    def test_dataset_consistent_loading(self, mock_dataframe):
        """Test that loading the same item multiple times gives consistent results."""
        dataset = LesionDataset(mock_dataframe, "clinical_score")

        # Load the same item multiple times
        data1, target1 = dataset[0]
        data2, target2 = dataset[0]

        # Should be identical
        assert torch.equal(data1, data2), "Data tensors should be identical"
        assert torch.equal(target1, target2), "Target tensors should be identical"

    def test_dataset_memory_efficiency(self, mock_dataframe):
        """Test that dataset doesn't preload all data (memory efficiency check)."""
        # This is more of a design verification test
        dataset = LesionDataset(mock_dataframe, "clinical_score")

        # Dataset should only store the DataFrame and target column name
        # It should not have preloaded tensors stored as attributes
        assert hasattr(dataset, "df")
        assert hasattr(dataset, "target_col")

        # Should not have preloaded data attributes
        assert not hasattr(dataset, "_preloaded_data")
        assert not hasattr(dataset, "_cached_tensors")
