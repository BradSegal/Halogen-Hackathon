"""
Unit tests for the lesion_analysis.data.loader module.

Tests include validation of data loading, schema enforcement,
and error handling for various edge cases.
"""

import pytest
import pandas as pd
from pydantic import ValidationError

from lesion_analysis.data.loader import LesionRecord, load_and_prepare_data


class TestLesionRecord:
    """Test cases for the LesionRecord Pydantic model."""

    def test_valid_record_creation(self, tmp_path):
        """Test Case 1: Happy Path - Valid record creation with existing file."""
        # Create a mock lesion file
        lesion_file = tmp_path / "lesion_001.nii.gz"
        lesion_file.write_text("mock lesion data")

        record_data = {
            "lesion_id": "lesion_001.nii.gz",
            "clinical_score": 5.5,
            "treatment_assignment": "Treatment",
            "outcome_score": 4.0,
            "lesion_filepath": lesion_file,
        }

        record = LesionRecord(**record_data)
        assert record.lesion_id == "lesion_001.nii.gz"
        assert record.clinical_score == 5.5
        assert record.treatment_assignment == "Treatment"
        assert record.outcome_score == 4.0
        assert record.lesion_filepath == lesion_file

    def test_na_string_to_none_conversion(self, tmp_path):
        """Test Case 4: 'N/A' string conversion to None."""
        # Create a mock lesion file
        lesion_file = tmp_path / "lesion_002.nii.gz"
        lesion_file.write_text("mock lesion data")

        record_data = {
            "lesion_id": "lesion_002.nii.gz",
            "clinical_score": 3.0,
            "treatment_assignment": "N/A",
            "outcome_score": 2.5,
            "lesion_filepath": lesion_file,
        }

        record = LesionRecord(**record_data)
        assert record.treatment_assignment is None

    def test_nan_to_none_conversion(self, tmp_path):
        """Test Case 4: NaN conversion to None."""
        # Create a mock lesion file
        lesion_file = tmp_path / "lesion_003.nii.gz"
        lesion_file.write_text("mock lesion data")

        record_data = {
            "lesion_id": "lesion_003.nii.gz",
            "clinical_score": 7.2,
            "treatment_assignment": "Control",
            "outcome_score": float("nan"),
            "lesion_filepath": lesion_file,
        }

        record = LesionRecord(**record_data)
        assert record.outcome_score is None

    def test_missing_lesion_file_validation_error(self, tmp_path):
        """Test Case 2: Missing lesion file raises ValueError."""
        # Create path to non-existent file
        non_existent_file = tmp_path / "missing_lesion.nii.gz"

        record_data = {
            "lesion_id": "missing_lesion.nii.gz",
            "clinical_score": 4.0,
            "treatment_assignment": "Treatment",
            "outcome_score": 3.0,
            "lesion_filepath": non_existent_file,
        }

        with pytest.raises(ValidationError) as exc_info:
            LesionRecord(**record_data)

        assert "File path does not exist" in str(exc_info.value)

    def test_malformed_clinical_score_validation_error(self, tmp_path):
        """Test Case 3: Non-numeric clinical_score raises ValidationError."""
        # Create a mock lesion file
        lesion_file = tmp_path / "lesion_004.nii.gz"
        lesion_file.write_text("mock lesion data")

        record_data = {
            "lesion_id": "lesion_004.nii.gz",
            "clinical_score": "invalid_score",  # Non-numeric value
            "treatment_assignment": "Control",
            "outcome_score": 2.0,
            "lesion_filepath": lesion_file,
        }

        with pytest.raises(ValidationError) as exc_info:
            LesionRecord(**record_data)

        # Validate that the error is related to type validation
        assert "Input should be a valid number" in str(exc_info.value)


class TestLoadAndPrepareData:
    """Test cases for the load_and_prepare_data function."""

    def test_happy_path_data_loading(self, tmp_path):
        """Test Case 1: Happy Path - Successful data loading and preparation."""
        # Create mock CSV file
        csv_path = tmp_path / "test_tasks.csv"
        csv_content = """lesion_id,clinical_score,treatment_assignment,outcome_score
lesion_001.nii.gz,5.0,Treatment,3.0
lesion_002.nii.gz,8.0,Control,9.0
lesion_003.nii.gz,2.0,N/A,N/A"""

        csv_path.write_text(csv_content)

        # Create mock lesion files
        lesions_dir = tmp_path / "lesions"
        lesions_dir.mkdir()
        (lesions_dir / "lesion_001.nii.gz").write_text("mock data 1")
        (lesions_dir / "lesion_002.nii.gz").write_text("mock data 2")
        (lesions_dir / "lesion_003.nii.gz").write_text("mock data 3")

        # Load and prepare data
        df = load_and_prepare_data(csv_path, lesions_dir)

        # Validate DataFrame shape and columns
        assert len(df) == 3
        expected_columns = [
            "lesion_id",
            "clinical_score",
            "treatment_assignment",
            "outcome_score",
            "lesion_filepath",
            "is_responder",
        ]
        assert all(col in df.columns for col in expected_columns)

        # Validate is_responder calculation
        # lesion_001: outcome (3.0) < clinical (5.0) → True
        # lesion_002: outcome (9.0) > clinical (8.0) → False
        # lesion_003: outcome is NaN → should be NaN/False
        assert (
            df.loc[df["lesion_id"] == "lesion_001.nii.gz", "is_responder"].iloc[0]
        )
        assert not (
            df.loc[df["lesion_id"] == "lesion_002.nii.gz", "is_responder"].iloc[0]
        )

        # Validate N/A handling - should be converted to pd.NA
        assert pd.isna(
            df.loc[df["lesion_id"] == "lesion_003.nii.gz", "treatment_assignment"].iloc[
                0
            ]
        )

    def test_missing_csv_file_error(self, tmp_path):
        """Test missing CSV file raises FileNotFoundError."""
        non_existent_csv = tmp_path / "missing.csv"
        lesions_dir = tmp_path / "lesions"
        lesions_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            load_and_prepare_data(non_existent_csv, lesions_dir)

        assert "Tasks CSV not found" in str(exc_info.value)

    def test_missing_lesions_directory_error(self, tmp_path):
        """Test missing lesions directory raises FileNotFoundError."""
        csv_path = tmp_path / "test_tasks.csv"
        csv_content = "lesion_id,clinical_score,treatment_assignment,outcome_score"
        csv_path.write_text(csv_content)

        non_existent_dir = tmp_path / "missing_lesions"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_and_prepare_data(csv_path, non_existent_dir)

        assert "Lesions directory not found" in str(exc_info.value)

    def test_missing_lesion_file_in_data_validation_error(self, tmp_path):
        """Test Case 2: CSV pointing to non-existent lesion file raises ValidationError."""
        # Create mock CSV with reference to missing file
        csv_path = tmp_path / "test_tasks.csv"
        csv_content = """lesion_id,clinical_score,treatment_assignment,outcome_score
lesion_001.nii.gz,5.0,Treatment,3.0
missing_lesion.nii.gz,8.0,Control,9.0"""
        csv_path.write_text(csv_content)

        # Create lesions directory with only one file
        lesions_dir = tmp_path / "lesions"
        lesions_dir.mkdir()
        (lesions_dir / "lesion_001.nii.gz").write_text("mock data 1")
        # missing_lesion.nii.gz is intentionally not created

        with pytest.raises(ValidationError) as exc_info:
            load_and_prepare_data(csv_path, lesions_dir)

        assert "File path does not exist" in str(exc_info.value)

    def test_malformed_data_validation_error(self, tmp_path):
        """Test Case 3: Malformed data in CSV raises ValidationError."""
        # Create mock CSV with invalid clinical_score
        csv_path = tmp_path / "test_tasks.csv"
        csv_content = """lesion_id,clinical_score,treatment_assignment,outcome_score
lesion_001.nii.gz,invalid_score,Treatment,3.0"""
        csv_path.write_text(csv_content)

        # Create lesions directory and file
        lesions_dir = tmp_path / "lesions"
        lesions_dir.mkdir()
        (lesions_dir / "lesion_001.nii.gz").write_text("mock data 1")

        with pytest.raises(ValidationError) as exc_info:
            load_and_prepare_data(csv_path, lesions_dir)

        assert "Input should be a valid number" in str(exc_info.value)
