# File: src/lesion_analysis/features/atlas.py

import joblib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


class AtlasFeatureExtractor:
    """
    Transforms NIfTI lesion maps into feature vectors based on an anatomical atlas.
    Each feature represents the mean lesion signal (lesion load) within an ROI.
    """

    def __init__(self, n_rois: int = 400, model_dir: Path = Path("models")):
        self.n_rois = n_rois
        self.atlas_name = f"schaefer_2018_{n_rois}rois"
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.masker_path = self.model_dir / f"{self.atlas_name}_masker.joblib"
        self.masker: NiftiLabelsMasker = None
        self.atlas_img: nib.Nifti1Image = None

    def fit(self):
        """
        Fetches the specified atlas, creates a NiftiLabelsMasker, and saves it to disk.
        This should be run once before training.
        """
        if self.masker_path.exists():
            print(f"Masker already exists at {self.masker_path}. Loading it.")
            self.masker = joblib.load(self.masker_path)
            # Still need to load the atlas image for inverse transform
            print("Fetching atlas for inverse transform...")
            atlas = fetch_atlas_schaefer_2018(n_rois=self.n_rois, resolution_mm=2)
            # atlas.maps can be either a filepath (str) or a Nifti1Image object (in tests)
            if isinstance(atlas.maps, str):
                self.atlas_img = nib.load(atlas.maps)
            else:
                self.atlas_img = atlas.maps
            return

        print("Fetching atlas and creating masker...")
        # Using the 2mm resolution version to match the data
        atlas = fetch_atlas_schaefer_2018(n_rois=self.n_rois, resolution_mm=2)
        # atlas.maps can be either a filepath (str) or a Nifti1Image object (in tests)
        if isinstance(atlas.maps, str):
            self.atlas_img = nib.load(atlas.maps)
        else:
            self.atlas_img = atlas.maps

        self.masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=False,  # We want lesion load, not z-scored values
            memory="nilearn_cache",
            verbose=5,
            n_jobs=1,  # Force serial execution to prevent multiprocessing deadlock
        )
        # Fit the masker (required before transform)
        self.masker.fit()
        joblib.dump(self.masker, self.masker_path)
        print(f"Masker saved to {self.masker_path}")

    def transform(self, df: pd.DataFrame, batch_size: int = 100) -> np.ndarray:
        """
        Transforms lesion files into ROI-based feature matrix using batch processing.

        Args:
            df: DataFrame with a 'lesion_filepath' column.
            batch_size: Number of files to process at once (default: 100).
                       Reduce if memory issues persist.

        Returns:
            A numpy array of shape (n_samples, n_rois).
        """
        if self.masker is None:
            if not self.masker_path.exists():
                raise FileNotFoundError("Masker not found. Please run .fit() first.")
            self.masker = joblib.load(self.masker_path)

        lesion_filepaths = df["lesion_filepath"].tolist()
        n_samples = len(lesion_filepaths)
        print(
            f"Transforming {n_samples} lesions into {self.n_rois} ROI features "
            f"(batch_size={batch_size})..."
        )

        # Process in batches to avoid memory issues
        all_features = []
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

        for i in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_filepaths = lesion_filepaths[start_idx:end_idx]

            # Transform this batch
            batch_features = self.masker.transform(batch_filepaths)
            all_features.append(batch_features)

        # Concatenate all batch results
        return np.vstack(all_features)

    @staticmethod
    def memory_efficient_inverse_transform(
        weights: np.ndarray, atlas_img: nib.Nifti1Image
    ) -> nib.Nifti1Image:
        """
        A memory-efficient implementation of nilearn's inverse_transform.

        Args:
            weights: A 1D numpy array of weights (e.g., model coefficients).
            atlas_img: The NIfTI image object of the atlas itself.

        Returns:
            A new NIfTI image with the weights projected back into 3D space.
        """
        if weights.ndim != 1:
            raise ValueError("Weights must be a 1D array.")

        atlas_data = atlas_img.get_fdata()
        output_map = np.zeros_like(atlas_data, dtype=np.float32)

        # The number of ROIs should match the length of the weights vector.
        # Atlas labels are typically 1-based, so they go from 1 to n_rois.
        num_rois = len(weights)

        for i in range(num_rois):
            roi_label = i + 1  # Atlas labels are 1-based
            weight = weights[i]
            output_map[atlas_data == roi_label] = weight

        return nib.Nifti1Image(output_map, atlas_img.affine, atlas_img.header)
