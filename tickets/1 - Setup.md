Of course. Here is the comprehensive development ticket for `CORE-01`. It is self-contained and provides all necessary detail for an external developer to implement, test, and complete the task without ambiguity.

---

### Ticket: CORE-01: Foundational Data Pipeline and Project Restructure

#### 1. Description
The current codebase suffers from critical architectural flaws that inhibit robust and scalable development. The data loading logic is duplicated across multiple files (`tasks.py`, `visualization.py`), is highly memory-inefficient by loading the entire dataset into RAM at once, and operates on an unsanitized data schema (column names with spaces).

This ticket mandates a complete refactor of the data ingestion and preparation process. We will establish a single, reliable source of truth for all data loading operations. This includes:
1.  Sanitizing the raw `tasks.csv` file.
2.  Creating a new, organized project structure for clarity.
3.  Implementing a strict data validation layer using Pydantic.
4.  Building a centralized, memory-efficient data loader.
5.  Creating a script to generate reproducible, stratified train/test data splits that all subsequent models will use.

This foundational work is a prerequisite for all other modeling tasks.

#### 2. Justification
This refactor is non-negotiable and is grounded in our core engineering principles:

*   **DRY (Don't Repeat Yourself):** The current system has at least two separate, inconsistent methods for loading lesion data. We will replace this with a single, canonical `load_and_prepare_data` function. This eliminates code redundancy and ensures that all parts of the application (modeling, analysis, visualization) operate on the exact same, correctly processed data.
*   **Strict Contracts:** We will introduce a `LesionRecord` Pydantic model. This model serves as an executable contract for our data, enforcing types, handling missing values (`N/A`, `NaN`), and validating data integrity at the earliest possible stage. This prevents a large class of potential bugs related to data quality.
*   **Fail Fast, Fail Loudly:** The new data loader will immediately verify the existence of all specified lesion files upon initialization. If any file is missing, the program will terminate with a clear, informative error message, rather than failing unpredictably deep inside a model training loop.
*   **Principle of Least Surprise (POLS):** We will sanitize the CSV column names to `snake_case` (e.g., `Clinical score` becomes `clinical_score`). This is standard practice and allows for easier, more predictable data access (e.g., `df.clinical_score`). Furthermore, we will create a single, explicit definition for the `is_responder` target variable, removing ambiguity for all downstream tasks.

#### 3. Implementation Plan

**Step 1: Sanitize `data/tasks.csv`**

1.  Open the file `data/tasks.csv` in a text editor or with a script.
2.  Modify the header row.
    *   Rename `Clinical score` to `clinical_score`.
    *   Rename `Treatment assignment` to `treatment_assignment`.
    *   Rename `Outcome score` to `outcome_score`.
3.  Save the modified file. The new header should be: `lesion_id,clinical_score,treatment_assignment,outcome_score`.

**Step 2: Restructure the `src/` Directory**

1.  **Delete** the obsolete file: `src/tasks.py`.
2.  Create the following new directory structure within `src/`:
    ```
    src/
    └── lesion_analysis/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   └── loader.py
        ├── features/
        │   └── __init__.py
        └── models/
            └── __init__.py
    ```
    *You can create empty `__init__.py` files to mark the directories as Python packages.*

**Step 3: Implement the Data Loader and Contracts**

1.  Create the file `src/lesion_analysis/data/loader.py`.
2.  Add the following code. This defines the Pydantic contract and the core data loading function.

    ```python
    # File: src/lesion_analysis/data/loader.py
    
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

**Step 4: Create the Data Splitting Script**

1.  Create a new file: `scripts/prepare_data.py`.
2.  Add the following code to create and save reproducible train/test splits.

    ```python
    # File: scripts/prepare_data.py
    
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from src.lesion_analysis.data.loader import load_and_prepare_data
    
    # Define project root relative to the script location
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    LESIONS_DIR = DATA_ROOT / "lesions"
    PROCESSED_DATA_DIR = DATA_ROOT / "processed"
    
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
    
    if __name__ == "__main__":
        main()
    ```
3.  Execute the script from your terminal: `python scripts/prepare_data.py`.

#### 4. Acceptance Criteria
*   [ ] The file `data/tasks.csv` has its column headers renamed to `snake_case`.
*   [ ] The file `src/tasks.py` has been deleted.
*   [ ] The directory `src/lesion_analysis/data/` exists and contains `loader.py`.
*   [ ] The `loader.py` module contains the `LesionRecord` Pydantic model and the `load_and_prepare_data` function as specified.
*   [ ] The script `scripts/prepare_data.py` exists and, when run, successfully generates `data/processed/train.csv` and `data/processed/test.csv`.
*   [ ] The generated `train.csv` and `test.csv` files contain the new columns: `lesion_filepath` and `is_responder`.

#### 5. Testing Requirements
*   **Unit Tests:**
    *   Create `tests/data/test_loader.py`.
    *   **Test Case 1: Happy Path:** Write a test using a mock CSV and a temporary directory with mock lesion files. Verify that `load_and_prepare_data` returns a DataFrame of the correct shape and that the `is_responder` column is calculated correctly.
    *   **Test Case 2: Missing Lesion File:** Write a test where the mock CSV points to a non-existent file. Assert that the `load_and_prepare_data` function raises a `ValueError` (from the Pydantic validator).
    *   **Test Case 3: Malformed Data:** Write a test with a mock CSV that contains a non-numeric value for `clinical_score`. Assert that Pydantic raises a `ValidationError`.
    *   **Test Case 4: 'N/A' and NaN Handling:** Verify that `treatment_assignment` with 'N/A' and `outcome_score` with `NaN` are correctly converted to `None` in the Pydantic model and `pd.NA` in the final DataFrame.

#### 6. Definition of Done
*   [ ] All Acceptance Criteria are met.
*   [ ] All required unit tests are written in `tests/data/test_loader.py` and pass.
*   [ ] The code has been formatted (`black`), linted (`ruff`), and type-checked (`mypy`) successfully.
*   [ ] The `CLAUDE.md` and project `README.md` files have been updated to reflect the new data loading and preparation procedure.
*   [ ] The code has been peer-reviewed and approved.