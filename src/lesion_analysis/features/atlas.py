# File: src/lesion_analysis/features/atlas.py

import joblib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from pathlib import Path

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

    def fit(self):
        """
        Fetches the specified atlas, creates a NiftiLabelsMasker, and saves it to disk.
        This should be run once before training.
        """
        if self.masker_path.exists():
            print(f"Masker already exists at {self.masker_path}. Loading it.")
            self.masker = joblib.load(self.masker_path)
            return

        print("Fetching atlas and creating masker...")
        # Using the 2mm resolution version to match the data
        atlas = fetch_atlas_schaefer_2018(n_rois=self.n_rois, resolution_mm=2)

        self.masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=False,  # We want lesion load, not z-scored values
            memory="nilearn_cache",
            verbose=0,
        )
        joblib.dump(self.masker, self.masker_path)
        print(f"Masker saved to {self.masker_path}")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms lesion files into ROI-based feature matrix.

        Args:
            df: DataFrame with a 'lesion_filepath' column.

        Returns:
            A numpy array of shape (n_samples, n_rois).
        """
        if self.masker is None:
            if not self.masker_path.exists():
                raise FileNotFoundError("Masker not found. Please run .fit() first.")
            self.masker = joblib.load(self.masker_path)

        lesion_filepaths = df["lesion_filepath"].tolist()
        print(f"Transforming {len(lesion_filepaths)} lesions into {self.n_rois} ROI features...")
        return self.masker.transform(lesion_filepaths)