# File: src/lesion_analysis/data/loader.py

from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, field_validator


class LesionRecord(BaseModel):
    """A Pydantic model for a single validated data record."""

    lesion_id: str
    clinical_score: float
    treatment_assignment: Optional[Literal["Control", "Treatment"]]
    outcome_score: Optional[float]
    lesion_filepath: Path

    @field_validator("treatment_assignment", mode="before")
    @classmethod
    def na_string_to_none(cls, v) -> Optional[str]:
        """Converts 'N/A' string and NaN values to a proper None value."""
        if v == "N/A":
            return None
        if pd.isna(v):
            return None
        return v

    @field_validator("outcome_score", mode="before")
    @classmethod
    def nan_to_none(cls, v) -> Optional[float]:
        """Converts pandas NaN to a proper None value."""
        return None if pd.isna(v) else v

    @field_validator("lesion_filepath")
    @classmethod
    def check_path_exists(cls, v: Path) -> Path:
        """Validates that the lesion file path actually exists."""
        if not v.exists():
            raise ValueError(f"File path does not exist: {v}")
        return v


def load_and_prepare_data(csv_path: Path, lesions_dir: Path) -> pd.DataFrame:
    """
    Loads the tasks CSV, validates its schema, enriches it with file paths,
    and adds derived columns for modeling.

    Args:
        csv_path: Path to the tasks.csv file.
        lesions_dir: Path to the directory containing the lesion NIfTI files.

    Returns:
        A pandas DataFrame with validated and prepared data.

    Raises:
        FileNotFoundError: If the CSV or lesions directory does not exist.
        pydantic.ValidationError: If the data does not conform to the schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Tasks CSV not found at {csv_path}")
    if not lesions_dir.exists():
        raise FileNotFoundError(f"Lesions directory not found at {lesions_dir}")

    df = pd.read_csv(csv_path)

    # --- Data Enrichment ---
    df["lesion_filepath"] = df["lesion_id"].apply(lambda x: lesions_dir / x)

    # --- Validation (Fail Fast, Fail Loudly) ---
    # Pydantic will iterate through the records and raise a comprehensive
    # ValidationError if any record is invalid (e.g., missing file, bad type).
    records = df.to_dict(orient="records")
    validated_records = [LesionRecord(**row) for row in records]  # type: ignore[misc]

    # --- Reconstruct DataFrame from validated data ---
    # This is critical - we use the clean, validated data from Pydantic
    df = pd.DataFrame([rec.model_dump() for rec in validated_records])

    # --- Feature Engineering ---
    # Define Responder Label (POLS)
    df["is_responder"] = (df["outcome_score"] < df["clinical_score"]).astype("boolean")

    # Clean treatment assignment for easier filtering
    df["treatment_assignment"] = df["treatment_assignment"].replace("N/A", pd.NA)

    return df
