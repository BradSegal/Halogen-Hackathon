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
