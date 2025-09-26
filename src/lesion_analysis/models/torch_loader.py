"""
PyTorch Dataset for loading lesion NIfTI files on-the-fly.

This module provides memory-efficient loading of neuroimaging data for deep learning
models, following the PyTorch Dataset interface.
"""

import torch
import pandas as pd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import Tuple


class LesionDataset(Dataset):
    """
    PyTorch Dataset for loading lesion NIfTI files on-the-fly.

    This dataset loads NIfTI files individually as requested by the DataLoader,
    which is highly memory-efficient for large neuroimaging datasets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing lesion metadata with columns:
        - lesion_filepath: path to NIfTI file
        - target column specified by target_col parameter
    target_col : str
        Name of the column containing target values

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Data tensor of shape (1, H, W, D) and target tensor (scalar)
    """

    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a single lesion map and its corresponding target.

        Parameters
        ----------
        idx : int
            Index of the sample to load

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Data tensor of shape (1, 91, 109, 91) and target scalar tensor
        """
        record = self.df.iloc[idx]
        filepath = record["lesion_filepath"]

        # Load image, ensure it has a channel dimension, and convert to tensor
        img = nib.load(filepath)
        data = torch.from_numpy(img.get_fdata(dtype=np.float32)).unsqueeze(  # type: ignore
            0
        )  # Shape: (1, 91, 109, 91)

        target_val = record[self.target_col]
        # Robustly handle missing values, default to 0.
        # This is safe because these samples will be filtered out by the training script anyway.
        if pd.isna(target_val):
            target_val = 0.0

        target = torch.tensor(target_val, dtype=torch.float32)

        return data, target
