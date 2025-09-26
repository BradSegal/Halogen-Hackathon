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
from typing import Tuple, Dict, Optional, Union


class LesionDataset(Dataset):
    """
    PyTorch Dataset for loading lesion NIfTI files on-the-fly.

    This dataset loads NIfTI files individually as requested by the DataLoader,
    which is highly memory-efficient for large neuroimaging datasets.

    Supports both single-task and multi-task learning modes:
    - Single-task: Specify target_col to return a single target value
    - Multi-task: Omit target_col to return dictionary of all targets

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing lesion metadata with columns:
        - lesion_filepath: path to NIfTI file
        - clinical_score: initial severity score
        - outcome_score: treatment outcome (may be NaN)
        - treatment_assignment: 'Control' or 'Treatment' (may be NaN)
    target_col : str, optional
        Name of the column containing target values for single-task mode.
        If None, returns dictionary of all targets for multi-task mode.

    Returns
    -------
    Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]
        Data tensor of shape (1, H, W, D) and either:
        - Single target tensor (if target_col specified)
        - Dictionary of target tensors (if target_col is None)
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Loads a single lesion map and its corresponding target(s).

        Parameters
        ----------
        idx : int
            Index of the sample to load

        Returns
        -------
        Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            Data tensor of shape (1, 91, 109, 91) and either:
            - Single target tensor (if target_col specified)
            - Dictionary of target tensors (if target_col is None)
        """
        record = self.df.iloc[idx]
        filepath = record["lesion_filepath"]

        # Load image, ensure it has a channel dimension, and convert to tensor
        img = nib.load(filepath)
        data = torch.from_numpy(img.get_fdata(dtype=np.float32)).unsqueeze(  # type: ignore
            0
        )  # Shape: (1, 91, 109, 91)

        # Single-task mode: return single target value
        if self.target_col is not None:
            target_val = record[self.target_col]
            # Robustly handle missing values, default to 0.
            # This is safe because these samples will be filtered out by the training script anyway.
            if pd.isna(target_val):
                target_val = 0.0
            target = torch.tensor(target_val, dtype=torch.float32)
            return data, target

        # Multi-task mode: return dictionary of targets
        else:
            # Encode treatment assignment 'W'. Control=0, Treatment=1.
            # If 'treatment_assignment' is NaN/None, default to 0 (it won't be used for outcome loss anyway).
            treatment_val = (
                1 if record.get("treatment_assignment") == "Treatment" else 0
            )

            targets = {
                "severity": torch.tensor(
                    record.get("clinical_score", np.nan), dtype=torch.float32
                ),
                "outcome": torch.tensor(
                    record.get("outcome_score", np.nan), dtype=torch.float32
                ),
                "treatment": torch.tensor(treatment_val, dtype=torch.float32),
            }

            return data, targets
