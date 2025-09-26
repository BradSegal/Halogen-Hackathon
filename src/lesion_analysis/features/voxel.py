import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from pathlib import Path
from tqdm import tqdm
import joblib


def downsample_and_flatten_lesions(
    df: pd.DataFrame,
    output_dir: Path,
    scale_factor: float = 0.25,  # Downsample to 1/4 resolution
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Loads lesion files from a DataFrame, downsamples them, flattens them into
    vectors, and caches the result.

    Args:
        df: DataFrame containing a 'lesion_filepath' column.
        output_dir: Directory to save the cached feature array.
        scale_factor: The factor by which to downsample each dimension.
        force_recompute: If True, recomputes features even if a cached file exists.

    Returns:
        A numpy array of shape (n_samples, n_flattened_voxels).
    """
    output_dir.mkdir(exist_ok=True)
    # Unique cache file name based on number of samples and scale factor
    cache_filename = f"voxel_features_{len(df)}_scale_{scale_factor}.joblib"
    cache_path = output_dir / cache_filename

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached features from {cache_path}")
        return joblib.load(cache_path)

    print("Computing voxel features...")
    feature_vectors = []

    # Determine target shape from the first image
    first_img = nib.load(df["lesion_filepath"].iloc[0])
    original_affine = first_img.affine
    target_affine = original_affine.copy()
    target_affine[:3, :3] *= 1 / scale_factor

    for filepath in tqdm(df["lesion_filepath"], desc="Downsampling lesions"):
        resampled_img = image.resample_img(
            filepath, target_affine=target_affine, interpolation="nearest"
        )
        feature_vectors.append(resampled_img.get_fdata().flatten())

    features = np.array(feature_vectors)
    joblib.dump(features, cache_path)
    print(f"Saved computed features to {cache_path}")

    return features
