import pytest
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil

from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions


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


def test_downsample_and_flatten_lesions_shape(mock_dataframe, temp_data_dir):
    """Test that the function returns correct shape array."""
    temp_dir, _ = temp_data_dir
    output_dir = temp_dir / "cache"

    scale_factor = 0.5  # Downsample to half resolution
    result = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=scale_factor
    )

    # Should have 3 samples (rows in DataFrame)
    assert result.shape[0] == 3

    # Should be flattened vectors
    assert len(result.shape) == 2

    # Each sample should be flattened downsampled voxels
    assert result.shape[1] > 0


def test_downsample_and_flatten_lesions_caching(mock_dataframe, temp_data_dir, capsys):
    """Test that caching works correctly."""
    temp_dir, _ = temp_data_dir
    output_dir = temp_dir / "cache"

    # First call - should compute features
    result1 = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=0.5
    )
    captured1 = capsys.readouterr()
    assert "Computing voxel features..." in captured1.out
    assert "Saved computed features to" in captured1.out

    # Second call - should load from cache
    result2 = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=0.5
    )
    captured2 = capsys.readouterr()
    assert "Loading cached features from" in captured2.out

    # Results should be identical
    np.testing.assert_array_equal(result1, result2)


def test_downsample_and_flatten_lesions_force_recompute(
    mock_dataframe, temp_data_dir, capsys
):
    """Test that force_recompute works correctly."""
    temp_dir, _ = temp_data_dir
    output_dir = temp_dir / "cache"

    # First call
    result1 = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=0.5
    )

    # Second call with force_recompute=True - should recompute
    result2 = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=0.5, force_recompute=True
    )
    captured2 = capsys.readouterr()
    assert "Computing voxel features..." in captured2.out

    # Results should still be identical (same input data)
    np.testing.assert_array_equal(result1, result2)


def test_downsample_and_flatten_lesions_creates_cache_file(
    mock_dataframe, temp_data_dir
):
    """Test that cache file is created."""
    temp_dir, _ = temp_data_dir
    output_dir = temp_dir / "cache"

    scale_factor = 0.5
    downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=scale_factor
    )

    # Check that cache file exists
    expected_cache_file = (
        output_dir / f"voxel_features_{len(mock_dataframe)}_scale_{scale_factor}.joblib"
    )
    assert expected_cache_file.exists()


def test_downsample_and_flatten_lesions_creates_output_dir(
    mock_dataframe, temp_data_dir
):
    """Test that output directory is created if it doesn't exist."""
    temp_dir, _ = temp_data_dir
    output_dir = temp_dir / "nonexistent_cache_dir"

    # Directory shouldn't exist initially
    assert not output_dir.exists()

    downsample_and_flatten_lesions(mock_dataframe, output_dir, scale_factor=0.5)

    # Directory should be created
    assert output_dir.exists()


def test_downsample_and_flatten_lesions_different_scale_factors(
    mock_dataframe, temp_data_dir
):
    """Test that different scale factors produce different cache files."""
    temp_dir, _ = temp_data_dir
    output_dir = temp_dir / "cache"

    # Test with two different scale factors
    result1 = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=0.5
    )
    result2 = downsample_and_flatten_lesions(
        mock_dataframe, output_dir, scale_factor=0.25
    )

    # Results should have different shapes (different downsampling)
    assert result1.shape != result2.shape

    # Should have different cache files
    cache1 = output_dir / f"voxel_features_{len(mock_dataframe)}_scale_0.5.joblib"
    cache2 = output_dir / f"voxel_features_{len(mock_dataframe)}_scale_0.25.joblib"

    assert cache1.exists()
    assert cache2.exists()
