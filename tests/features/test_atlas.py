import pytest
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.lesion_analysis.features.atlas import AtlasFeatureExtractor


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with dummy NIfTI files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create dummy 3D lesion data (small size for testing)
    shape = (10, 12, 8)  # Much smaller than real data for faster testing
    affine = np.eye(4)

    # Create multiple dummy lesion files
    filepaths = []
    for i in range(3):
        # Random binary lesion data
        data = np.random.randint(0, 2, size=shape, dtype=np.uint8)
        img = nib.Nifti1Image(data, affine)
        filepath = temp_dir / f"lesion_{i:04d}.nii.gz"
        nib.save(img, str(filepath))
        filepaths.append(str(filepath))

    yield temp_dir, filepaths

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_dataframe(temp_data_dir):
    """Create mock DataFrame with lesion filepaths."""
    temp_dir, filepaths = temp_data_dir

    df = pd.DataFrame(
        {
            "lesion_id": [f"lesion_{i:04d}.nii.gz" for i in range(3)],
            "lesion_filepath": filepaths,
            "clinical_score": [1.0, 2.0, 3.0],
        }
    )

    return df


@pytest.fixture
def mock_atlas():
    """Create mock atlas data for testing."""
    # Create a simple atlas with 4 ROIs (much smaller than real 400 ROIs)
    shape = (10, 12, 8)
    affine = np.eye(4)

    # Create atlas data with 4 distinct regions
    atlas_data = np.zeros(shape, dtype=np.int32)
    atlas_data[0:5, :, :] = 1  # ROI 1
    atlas_data[5:10, 0:6, :] = 2  # ROI 2
    atlas_data[5:10, 6:12, 0:4] = 3  # ROI 3
    atlas_data[5:10, 6:12, 4:8] = 4  # ROI 4

    atlas_img = nib.Nifti1Image(atlas_data, affine)

    # Mock atlas object
    mock_atlas = MagicMock()
    mock_atlas.maps = atlas_img

    return mock_atlas


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extractor_fit(mock_fetch, temp_data_dir, mock_atlas):
    """Test that fit method creates and saves masker correctly."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    extractor = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    extractor.fit()

    # Check that fetch was called with correct parameters
    mock_fetch.assert_called_once_with(n_rois=4, resolution_mm=2)

    # Check that masker was created and saved
    assert extractor.masker is not None
    assert extractor.masker_path.exists()


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extractor_fit_loads_existing(
    mock_fetch, temp_data_dir, mock_atlas, capsys
):
    """Test that fit method loads existing masker if it exists."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    extractor = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)

    # First fit - should create masker
    extractor.fit()
    captured1 = capsys.readouterr()
    assert "Fetching atlas and creating masker..." in captured1.out

    # Second fit - should load existing masker
    extractor2 = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    extractor2.fit()
    captured2 = capsys.readouterr()
    assert "Masker already exists" in captured2.out


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extractor_transform_shape(
    mock_fetch, mock_dataframe, temp_data_dir, mock_atlas
):
    """Test that transform method returns correct shape array."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    extractor = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    extractor.fit()

    result = extractor.transform(mock_dataframe)

    # Should have 3 samples (rows in DataFrame) and 4 ROI features
    assert result.shape == (3, 4)

    # All values should be non-negative (lesion loads)
    assert np.all(result >= 0)


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extractor_transform_without_fit_loads_masker(
    mock_fetch, mock_dataframe, temp_data_dir, mock_atlas
):
    """Test that transform method loads masker if not already loaded."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    # First create and save masker
    extractor1 = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    extractor1.fit()

    # Create new extractor and transform without calling fit first
    extractor2 = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    result = extractor2.transform(mock_dataframe)

    # Should work correctly and return proper shape
    assert result.shape == (3, 4)
    assert extractor2.masker is not None


def test_atlas_feature_extractor_transform_without_masker_raises_error(
    mock_dataframe, temp_data_dir
):
    """Test that transform raises error when no masker exists."""
    temp_dir, _ = temp_data_dir

    extractor = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)

    with pytest.raises(
        FileNotFoundError, match="Masker not found. Please run .fit\\(\\) first."
    ):
        extractor.transform(mock_dataframe)


def test_atlas_feature_extractor_creates_model_dir(temp_data_dir):
    """Test that model directory is created if it doesn't exist."""
    temp_dir, _ = temp_data_dir
    model_dir = temp_dir / "nonexistent_models_dir"

    # Directory shouldn't exist initially
    assert not model_dir.exists()

    AtlasFeatureExtractor(n_rois=4, model_dir=model_dir)

    # Directory should be created during initialization
    assert model_dir.exists()


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extractor_different_n_rois(
    mock_fetch, temp_data_dir, mock_atlas
):
    """Test that different n_rois values create different masker files."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    # Create extractors with different n_rois
    extractor1 = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    extractor1.fit()

    extractor2 = AtlasFeatureExtractor(n_rois=8, model_dir=temp_dir)
    extractor2.fit()

    # Should have different masker file names
    assert extractor1.masker_path != extractor2.masker_path
    assert extractor1.masker_path.exists()
    assert extractor2.masker_path.exists()


def test_memory_efficient_inverse_transform():
    """Test the memory-efficient inverse transform method."""
    # Create a small dummy 3D atlas (4x4x4 with 3 regions)
    shape = (4, 4, 4)
    affine = np.eye(4)

    # Create atlas data with 3 distinct regions
    atlas_data = np.zeros(shape, dtype=np.int32)
    atlas_data[0:2, 0:2, :] = 1  # ROI 1
    atlas_data[2:4, 0:2, :] = 2  # ROI 2
    atlas_data[:, 2:4, :] = 3  # ROI 3

    atlas_img = nib.Nifti1Image(atlas_data, affine)

    # Create dummy weight vector
    weights = np.array([10.0, 20.0, 30.0])

    # Call the function
    output_img = AtlasFeatureExtractor.memory_efficient_inverse_transform(
        weights, atlas_img
    )

    # Get the output data
    output_data = output_img.get_fdata()

    # Assert that voxels in each region have the correct values
    # ROI 1 should have value 10
    assert np.all(output_data[atlas_data == 1] == 10.0)
    # ROI 2 should have value 20
    assert np.all(output_data[atlas_data == 2] == 20.0)
    # ROI 3 should have value 30
    assert np.all(output_data[atlas_data == 3] == 30.0)
    # Background (ROI 0) should have value 0
    assert np.all(output_data[atlas_data == 0] == 0.0)

    # Check that affine and header are preserved
    assert np.array_equal(output_img.affine, atlas_img.affine)
    assert output_img.header == atlas_img.header


def test_memory_efficient_inverse_transform_error_on_non_1d():
    """Test that memory_efficient_inverse_transform raises error for non-1D weights."""
    # Create a dummy atlas
    shape = (4, 4, 4)
    affine = np.eye(4)
    atlas_data = np.zeros(shape, dtype=np.int32)
    atlas_img = nib.Nifti1Image(atlas_data, affine)

    # Create 2D weight array (should cause error)
    weights = np.array([[10.0, 20.0], [30.0, 40.0]])

    with pytest.raises(ValueError, match="Weights must be a 1D array"):
        AtlasFeatureExtractor.memory_efficient_inverse_transform(weights, atlas_img)


def test_memory_efficient_inverse_transform_with_zeros():
    """Test memory_efficient_inverse_transform with zero weights."""
    # Create a small dummy atlas
    shape = (3, 3, 3)
    affine = np.eye(4)
    atlas_data = np.ones(shape, dtype=np.int32)  # All voxels are ROI 1
    atlas_img = nib.Nifti1Image(atlas_data, affine)

    # Create weights with zero
    weights = np.array([0.0])

    output_img = AtlasFeatureExtractor.memory_efficient_inverse_transform(
        weights, atlas_img
    )
    output_data = output_img.get_fdata()

    # All voxels should be zero
    assert np.all(output_data == 0.0)


def test_memory_efficient_inverse_transform_with_negative_weights():
    """Test memory_efficient_inverse_transform with negative weights."""
    # Create a small dummy atlas
    affine = np.eye(4)
    atlas_data = np.array([[[1, 2], [3, 1]], [[2, 3], [1, 2]]], dtype=np.int32)
    atlas_img = nib.Nifti1Image(atlas_data, affine)

    # Create weights with negative values
    weights = np.array([-5.0, 10.0, -15.0])

    output_img = AtlasFeatureExtractor.memory_efficient_inverse_transform(
        weights, atlas_img
    )
    output_data = output_img.get_fdata()

    # Check that negative weights are correctly applied
    assert np.all(output_data[atlas_data == 1] == -5.0)
    assert np.all(output_data[atlas_data == 2] == 10.0)
    assert np.all(output_data[atlas_data == 3] == -15.0)


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_img_stored_after_fit(mock_fetch, temp_data_dir, mock_atlas):
    """Test that atlas_img is stored after calling fit."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    extractor = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)

    # Initially atlas_img should be None
    assert extractor.atlas_img is None

    extractor.fit()

    # After fit, atlas_img should be loaded
    assert extractor.atlas_img is not None
    assert isinstance(extractor.atlas_img, nib.Nifti1Image)


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extractor_batch_consistency(
    mock_fetch, mock_dataframe, temp_data_dir, mock_atlas
):
    """Test that different batch sizes produce identical results."""
    temp_dir, _ = temp_data_dir
    mock_fetch.return_value = mock_atlas

    extractor = AtlasFeatureExtractor(n_rois=4, model_dir=temp_dir)
    extractor.fit()

    # Transform with different batch sizes
    result_batch1 = extractor.transform(mock_dataframe, batch_size=1)
    result_batch2 = extractor.transform(mock_dataframe, batch_size=2)
    result_batch3 = extractor.transform(mock_dataframe, batch_size=3)
    result_batch100 = extractor.transform(mock_dataframe, batch_size=100)

    # All results should be identical
    assert np.allclose(result_batch1, result_batch2)
    assert np.allclose(result_batch1, result_batch3)
    assert np.allclose(result_batch1, result_batch100)

    # Shape should be consistent
    assert result_batch1.shape == (3, 4)
    assert result_batch2.shape == (3, 4)
    assert result_batch3.shape == (3, 4)
    assert result_batch100.shape == (3, 4)
