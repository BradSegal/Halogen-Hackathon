# Implementation Plan for CORE-01: Foundational Data Pipeline and Project Restructure

## 1. High-Level Strategy

This ticket requires a complete refactor of the data ingestion and preparation system to establish a single, reliable, memory-efficient data loading pipeline. The implementation will replace duplicated data loading logic with a validated, modular architecture that serves as the foundation for all subsequent ML tasks.

**Core Philosophy**: Fail fast with clear errors, use strict contracts, maintain DRY principles, and follow predictable naming conventions (POLS).

## 2. Pre-Implementation Dependencies

### Required Dependency Addition:
- **File Path**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/pyproject.toml`
- **Change**: Add `"pydantic"` to dependencies list
- **Rationale**: Required for data validation models

## 3. Step-by-Step Implementation Plan

### Step 1: Data Schema Sanitization
**Objective**: Clean CSV column headers to follow snake_case convention

**File to Modify**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/data/tasks.csv`
**Operation**: Update header row from:
```
lesion_id,Clinical score,Treatment assignment,Outcome score
```
to:
```
lesion_id,clinical_score,treatment_assignment,outcome_score
```

**Validation**: Verify 4119 data rows remain unchanged, only header modified

### Step 2: Remove Obsolete Code
**File to Delete**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/src/tasks.py`
**Rationale**: This monolithic file contains duplicated, memory-inefficient data loading logic that will be replaced by the modular structure.

### Step 3: Create New Project Structure

**Directories to Create**:
```
src/lesion_analysis/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── loader.py
├── features/
│   └── __init__.py
└── models/
    └── __init__.py
```

**Implementation Details**:
- All `__init__.py` files: Empty files to mark Python packages
- Structure enables clear separation of concerns: data loading, feature engineering, and modeling

### Step 4: Implement Core Data Models and Validation

**File to Create**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/src/lesion_analysis/data/loader.py`

#### 4.1 Pydantic Data Model
```python
from pathlib import Path
from typing import List, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field, validator

class LesionRecord(BaseModel):
    """A Pydantic model for a single validated data record."""
    lesion_id: str
    clinical_score: float
    treatment_assignment: Optional[Literal["Control", "Treatment"]]
    outcome_score: Optional[float]
    lesion_filepath: Path

    @validator("treatment_assignment")
    def na_string_to_none(cls, v: str) -> Optional[str]:
        """Converts 'N/A' string to a proper None value."""
        return None if v == "N/A" else v

    @validator("outcome_score")
    def nan_to_none(cls, v: float) -> Optional[float]:
        """Converts pandas NaN to a proper None value."""
        return None if pd.isna(v) else v

    @validator("lesion_filepath")
    def check_path_exists(cls, v: Path) -> Path:
        """Validates that the lesion file path actually exists."""
        if not v.exists():
            raise ValueError(f"File path does not exist: {v}")
        return v
```

**Key Features**:
- **Type Safety**: Enforces exact types for all fields
- **Null Handling**: Converts inconsistent "N/A" strings to proper None values
- **File Validation**: Fails immediately if lesion files are missing
- **Clear Contracts**: Self-documenting data expectations

#### 4.2 Core Data Loading Function
```python
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
    _ = [LesionRecord(**row) for row in df.to_dict(orient="records")]

    # --- Feature Engineering ---
    # Define Responder Label (POLS)
    df["is_responder"] = (df["outcome_score"] < df["clinical_score"]).astype('boolean')

    # Clean treatment assignment for easier filtering
    df['treatment_assignment'] = df['treatment_assignment'].replace('N/A', pd.NA)

    return df
```

**Key Features**:
- **Fail-Fast Validation**: Validates ALL records before returning any data
- **Memory Efficiency**: Loads only CSV metadata, not actual lesion arrays
- **Feature Engineering**: Adds derived `is_responder` column for classification tasks
- **Clear Error Messages**: Specific file paths and validation failures

### Step 5: Create Data Preparation Script

**File to Create**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/scripts/prepare_data.py`

#### 5.1 Directory Structure Creation
```python
# Define project root relative to the script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
LESIONS_DIR = DATA_ROOT / "lesions"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
```

#### 5.2 Main Processing Function
```python
def main():
    """
    Main function to execute the data preparation and splitting process.
    """
    print("Starting data preparation...")
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data(
        csv_path=DATA_ROOT / "tasks.csv",
        lesions_dir=LESIONS_DIR
    )
    print(f"Loaded and validated {len(df)} records.")

    # Stratify on treatment_assignment, filling NaNs for stratification to work
    stratify_col = df['treatment_assignment'].fillna('None')

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col
    )
    print(f"Data split into {len(train_df)} training and {len(test_df)} test samples.")

    # Save splits
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print("Data preparation complete.")
```

**Key Features**:
- **Reproducible Splits**: Fixed random_state=42 for consistent train/test splits
- **Stratified Sampling**: Maintains treatment assignment distribution in splits
- **Output Management**: Creates `data/processed/` directory structure
- **Progress Reporting**: Clear console output for each step

### Step 6: Comprehensive Unit Testing

**File to Create**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/tests/data/test_loader.py`

#### 6.1 Test Class Structure
```python
import pytest
import tempfile
import pandas as pd
from pathlib import Path
from pydantic import ValidationError
import numpy as np

class TestLesionDataLoader:
    """Comprehensive tests for the lesion data loading system."""
```

#### 6.2 Required Test Cases

**Test Case 1: Happy Path**
- **Function**: `test_load_and_prepare_data_happy_path()`
- **Setup**: Create mock CSV with valid data and temporary NIfTI files
- **Assertions**:
  - Returned DataFrame has correct shape (4119 rows)
  - Contains all required columns: `lesion_id`, `clinical_score`, `treatment_assignment`, `outcome_score`, `lesion_filepath`, `is_responder`
  - `is_responder` calculated correctly: `outcome_score < clinical_score`
  - `treatment_assignment` "N/A" values converted to `pd.NA`

**Test Case 2: Missing Lesion File Validation**
- **Function**: `test_missing_lesion_file_raises_error()`
- **Setup**: Create CSV pointing to non-existent lesion file
- **Assertions**:
  - `load_and_prepare_data()` raises `ValueError`
  - Error message contains specific missing file path
  - Error occurs during Pydantic validation phase

**Test Case 3: Malformed Data Validation**
- **Function**: `test_malformed_data_raises_validation_error()`
- **Setup**: Create CSV with non-numeric `clinical_score` (e.g., "invalid")
- **Assertions**:
  - Pydantic raises `ValidationError`
  - Error message specifies invalid field and value
  - No partial data is returned

**Test Case 4: N/A and NaN Handling**
- **Function**: `test_na_and_nan_handling()`
- **Setup**: Create CSV with "N/A" strings and NaN float values
- **Assertions**:
  - `treatment_assignment` "N/A" becomes `None` in Pydantic model
  - `outcome_score` NaN becomes `None` in Pydantic model
  - Final DataFrame uses `pd.NA` for consistency

#### 6.3 Test Data Management
- **Fixtures**: Use `@pytest.fixture` for creating temporary test data
- **Cleanup**: Automatic cleanup of temporary files and directories
- **Mock Data**: Small, focused datasets (5-10 records) for fast execution

### Step 7: Integration Testing

**File to Create**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/tests/integration/test_data_pipeline.py`

**Test Scenario**: `test_end_to_end_data_preparation()`
- **Scope**: Full pipeline from CSV loading through train/test split generation
- **Setup**: Use actual `data/tasks.csv` with subset of lesion files
- **Validation**:
  - `scripts/prepare_data.py` executes without errors
  - Generated `train.csv` and `test.csv` have correct schemas
  - Split proportions are correct (80/20)
  - Stratification maintains treatment distribution

## 4. Components to Modify

### Existing Files Requiring Updates:

**File**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/pyproject.toml`
- **Change**: Add `"pydantic"` to dependencies
- **Rationale**: Required for data validation models

## 5. Components to Delete

**File**: `/mnt/c/Users/bradl/PycharmProjects/Halogen Hackathon/.agent_work/worktrees/CORE-01/src/tasks.py`
- **Rationale**: Contains obsolete, memory-inefficient data loading logic

## 6. Risk Mitigation Strategies

### Memory Management Risk
- **Risk**: Large datasets could still cause memory issues
- **Mitigation**: Implement lazy loading pattern; load lesion arrays only when needed
- **Monitoring**: Add memory usage logging during development

### Validation Performance Risk
- **Risk**: Pydantic validation might slow down data loading
- **Mitigation**: Profile validation overhead; implement caching for repeated loads
- **Acceptable**: Validation cost is front-loaded and prevents expensive debugging later

### Integration Disruption Risk
- **Risk**: Existing visualization/analysis code depends on old data formats
- **Mitigation**: Maintain consistent pandas DataFrame interface; update dependent modules gradually

## 7. Success Criteria

### Technical Success Metrics:
1. **Memory Usage**: Data loading uses <100MB RAM (down from 1.5GB+)
2. **Error Clarity**: All validation errors include specific file paths and field names
3. **Code Deduplication**: Single `load_and_prepare_data()` function replaces 2+ implementations
4. **Test Coverage**: >95% line coverage for all data loading code

### Functional Success Metrics:
1. **Data Integrity**: 100% of 4119 records validated successfully
2. **Reproducibility**: Identical train/test splits across multiple runs
3. **Schema Consistency**: All column names follow snake_case convention

### Process Success Metrics:
1. **Development Velocity**: New models can be developed without debugging data loading issues
2. **Onboarding**: New developers can understand data loading in <30 minutes
3. **Maintenance**: Code changes require updating only single data loading module

This implementation plan provides a clear, step-by-step roadmap for establishing robust data infrastructure that will serve as the foundation for all subsequent ML development work.