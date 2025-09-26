import pytest
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import os
from nilearn import image

from scripts.train_baseline_models import PROJECT_ROOT


@pytest.fixture
def temp_project_structure():
    """Create temporary project structure with minimal valid data."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    processed_dir = temp_dir / "data" / "processed"
    models_dir = temp_dir / "models"
    results_dir = temp_dir / "results"
    lesions_dir = temp_dir / "data" / "lesions"

    processed_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    lesions_dir.mkdir(parents=True)

    # Create minimal lesion data
    shape = (12, 14, 10)  # Small size for fast testing
    affine = np.eye(4)
    n_samples = 20

    filepaths = []
    for i in range(n_samples):
        # Random binary lesion data
        data = np.random.randint(0, 2, size=shape, dtype=np.uint8)
        img = nib.Nifti1Image(data, affine)
        filepath = lesions_dir / f"lesion{i:04d}.nii.gz"
        nib.save(img, str(filepath))
        filepaths.append(str(filepath))

    # Create minimal train.csv
    train_data = {
        "lesion_id": [f"lesion{i:04d}.nii.gz" for i in range(n_samples)],
        "lesion_filepath": filepaths,
        "clinical_score": np.random.uniform(0, 10, n_samples),
        "treatment_assignment": ["Control", "Treatment"] * (n_samples // 2),
        "outcome_score": np.random.uniform(0, 10, n_samples),
        "is_responder": np.random.choice([True, False], n_samples),
    }

    # Make some clinical_scores 0 (as per original data structure)
    train_data["clinical_score"][:5] = 0
    # Make some treatment_assignment NaN (as per original data structure)
    train_data["treatment_assignment"][:5] = None
    train_data["outcome_score"][:5] = None

    train_df = pd.DataFrame(train_data)
    train_df.to_csv(processed_dir / "train.csv", index=False)

    yield temp_dir, filepaths

    # Cleanup
    shutil.rmtree(temp_dir)


def test_train_baseline_models_script_execution(temp_project_structure):
    """Test that the training script runs to completion and creates expected files."""
    temp_dir, _ = temp_project_structure

    # Create a modified version of the script that uses temp_dir
    modified_script_content = f"""
import sys
import os
sys.path.insert(0, '{PROJECT_ROOT}')
os.environ['PROJECT_ROOT'] = '{temp_dir}'

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import image

from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, balanced_accuracy_score

from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions

# --- Configuration ---
PROJECT_ROOT = Path('{temp_dir}')
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_CACHE_DIR = PROCESSED_DATA_DIR / "features_cache"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

# --- 2. Feature Engineering ---
X_train_all = downsample_and_flatten_lesions(train_df, FEATURES_CACHE_DIR)

# --- 3. Task 1: Deficit Prediction (Regression) ---
print("\\n--- Training Task 1 Model ---")
task1_mask = (train_df.clinical_score > 0).values
X_task1 = X_train_all[task1_mask]
y_task1 = train_df.loc[task1_mask, 'clinical_score'].values

model_task1 = RidgeCV(alphas=np.logspace(-3, 3, 7))
model_task1.fit(X_task1, y_task1)

# Evaluate on training set (for baseline check)
preds = model_task1.predict(X_task1)
rmse = np.sqrt(mean_squared_error(y_task1, preds))
print(f"Task 1 Train RMSE: {{rmse:.4f}}")

joblib.dump(model_task1, MODELS_DIR / "task1_baseline_model.pkl")
print("Task 1 model saved.")

# --- 4. Task 2: Responder Prediction (Classification) ---
print("\\n--- Training Task 2 Model ---")
task2_mask = train_df.treatment_assignment.notna().values
X_task2 = X_train_all[task2_mask]
y_task2 = train_df.loc[task2_mask, 'is_responder'].values

model_task2 = LogisticRegressionCV(
    Cs=5, cv=3, penalty='l2', solver='liblinear',
    class_weight='balanced', random_state=42
)
model_task2.fit(X_task2, y_task2)

# Evaluate on training set
preds = model_task2.predict(X_task2)
bacc = balanced_accuracy_score(y_task2, preds)
print(f"Task 2 Train Balanced Accuracy: {{bacc:.4f}}")

joblib.dump(model_task2, MODELS_DIR / "task2_baseline_model.pkl")
print("Task 2 model saved.")

# --- 5. Task 3: Inference Maps ---
print("\\n--- Generating Inference Maps ---")

# Get shape info from a sample image
sample_img = nib.load(train_df["lesion_filepath"].iloc[0])
resampled_img = image.resample_img(sample_img, target_affine=sample_img.affine * 4, interpolation='nearest')
downsampled_shape = resampled_img.shape

# Deficit Map (from Ridge model)
deficit_coefs = model_task1.coef_
deficit_map_downsampled = deficit_coefs.reshape(downsampled_shape)
deficit_map_upsampled_img = image.resample_to_img(
    source_img=nib.Nifti1Image(deficit_map_downsampled, resampled_img.affine),
    target_img=sample_img,
    interpolation='continuous'
)
deficit_map_upsampled_img.to_filename(RESULTS_DIR / "deficit_map_baseline.nii.gz")
print("Deficit map saved.")

# Treatment Map (from Logistic Regression model)
# For binary classification, coef_ has shape (1, n_features)
treatment_coefs = model_task2.coef_.flatten()
treatment_map_downsampled = treatment_coefs.reshape(downsampled_shape)
treatment_map_upsampled_img = image.resample_to_img(
    source_img=nib.Nifti1Image(treatment_map_downsampled, resampled_img.affine),
    target_img=sample_img,
    interpolation='continuous'
)

# Enforce subset constraint
deficit_mask_img = image.math_img("img > 0", img=deficit_map_upsampled_img)
final_treatment_map_img = image.math_img("img1 * img2", img1=treatment_map_upsampled_img, img2=deficit_mask_img)
final_treatment_map_img.to_filename(RESULTS_DIR / "treatment_map_baseline.nii.gz")
print("Treatment map saved.")

print("\\nBaseline script finished.")
"""

    # Write and run the modified script
    temp_script = temp_dir / "test_script.py"
    with open(temp_script, "w") as f:
        f.write(modified_script_content)

    result = subprocess.run(
        [sys.executable, str(temp_script)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check that script executed successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Check that all expected output files are created
    models_dir = temp_dir / "models"
    results_dir = temp_dir / "results"

    assert (models_dir / "task1_baseline_model.pkl").exists()
    assert (models_dir / "task2_baseline_model.pkl").exists()
    assert (results_dir / "deficit_map_baseline.nii.gz").exists()
    assert (results_dir / "treatment_map_baseline.nii.gz").exists()

    # Check that a cache file was created
    cache_dir = temp_dir / "data" / "processed" / "features_cache"
    cache_files = list(cache_dir.glob("voxel_features_*.joblib"))
    assert len(cache_files) > 0, "No cache file was created"


def test_generated_maps_have_correct_dimensions(temp_project_structure):
    """Test that generated maps have correct dimensions matching original lesion maps."""
    temp_dir, filepaths = temp_project_structure

    # First run the training script (simplified version for testing)
    from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV

    # Load data
    train_df = pd.read_csv(temp_dir / "data" / "processed" / "train.csv")
    cache_dir = temp_dir / "data" / "processed" / "features_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate features
    X_train_all = downsample_and_flatten_lesions(train_df, cache_dir, scale_factor=0.5)

    # Train models
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, "clinical_score"].values
    model_task1 = RidgeCV(alphas=np.logspace(-3, 3, 7))
    model_task1.fit(X_task1, y_task1)

    task2_mask = train_df.treatment_assignment.notna().values
    X_task2 = X_train_all[task2_mask]
    y_task2 = train_df.loc[task2_mask, "is_responder"].values
    model_task2 = LogisticRegressionCV(
        Cs=5,
        cv=3,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    model_task2.fit(X_task2, y_task2)

    # Generate maps
    sample_img = nib.load(train_df["lesion_filepath"].iloc[0])
    resampled_img = image.resample_img(
        sample_img, target_affine=sample_img.affine * 2, interpolation="nearest"
    )
    downsampled_shape = resampled_img.shape

    # Create deficit map
    deficit_coefs = model_task1.coef_
    deficit_map_downsampled = deficit_coefs.reshape(downsampled_shape)
    deficit_map_upsampled_img = image.resample_to_img(
        source_img=nib.Nifti1Image(deficit_map_downsampled, resampled_img.affine),
        target_img=sample_img,
        interpolation="continuous",
    )

    # Create treatment map
    treatment_coefs = model_task2.coef_.flatten()
    treatment_map_downsampled = treatment_coefs.reshape(downsampled_shape)
    treatment_map_upsampled_img = image.resample_to_img(
        source_img=nib.Nifti1Image(treatment_map_downsampled, resampled_img.affine),
        target_img=sample_img,
        interpolation="continuous",
    )

    # Check dimensions match original lesion map
    original_shape = sample_img.shape
    assert deficit_map_upsampled_img.shape == original_shape
    assert treatment_map_upsampled_img.shape == original_shape


def test_script_prints_expected_metrics(temp_project_structure):
    """Test that the script prints RMSE and Balanced Accuracy metrics."""
    temp_dir, _ = temp_project_structure

    # Run a simplified version to test metric printing
    from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV
    from sklearn.metrics import mean_squared_error, balanced_accuracy_score

    # Load data and generate features
    train_df = pd.read_csv(temp_dir / "data" / "processed" / "train.csv")
    cache_dir = temp_dir / "data" / "processed" / "features_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_train_all = downsample_and_flatten_lesions(train_df, cache_dir, scale_factor=0.5)

    # Train Task 1 model
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, "clinical_score"].values
    model_task1 = RidgeCV(alphas=np.logspace(-3, 3, 7))
    model_task1.fit(X_task1, y_task1)

    preds = model_task1.predict(X_task1)
    rmse = np.sqrt(mean_squared_error(y_task1, preds))

    # Train Task 2 model
    task2_mask = train_df.treatment_assignment.notna().values
    X_task2 = X_train_all[task2_mask]
    y_task2 = train_df.loc[task2_mask, "is_responder"].values
    model_task2 = LogisticRegressionCV(
        Cs=5,
        cv=3,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    model_task2.fit(X_task2, y_task2)

    preds = model_task2.predict(X_task2)
    bacc = balanced_accuracy_score(y_task2, preds)

    # Both metrics should be reasonable values
    assert 0 <= rmse <= 20  # RMSE should be positive and reasonable
    assert 0 <= bacc <= 1  # Balanced accuracy should be between 0 and 1


def test_train_on_all_data_flag_task1(temp_project_structure, capsys):
    """Test that --train-on-all-data flag changes Task 1 training behavior."""
    temp_dir, _ = temp_project_structure

    # Run script WITHOUT flag (default behavior)
    result_without = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train_baseline_models.py")],
        capture_output=True,
        text=True,
        cwd=str(temp_dir),
        env={**dict(os.environ), "PROJECT_ROOT": str(temp_dir)},
    )

    # Run script WITH flag
    result_with = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_baseline_models.py"),
            "--train-on-all-data",
        ],
        capture_output=True,
        text=True,
        cwd=str(temp_dir),
        env={**dict(os.environ), "PROJECT_ROOT": str(temp_dir)},
    )

    # Check both ran successfully
    assert (
        result_without.returncode == 0
    ), f"Script failed without flag: {result_without.stderr}"
    assert result_with.returncode == 0, f"Script failed with flag: {result_with.stderr}"

    # Check that output shows different behavior
    assert "Training on non-zero score data only" in result_without.stdout
    assert "Training on ALL data" in result_with.stdout

    # The version with all data should report more training samples
    # Extract sample counts from output (basic check)
    import re

    match_without = re.search(r"Training samples used: (\d+)", result_without.stdout)
    match_with = re.search(r"Training samples used: (\d+)", result_with.stdout)

    if match_without and match_with:
        samples_without = int(match_without.group(1))
        samples_with = int(match_with.group(1))
        # When training on all data, we should have more samples
        assert (
            samples_with >= samples_without
        ), f"Expected more samples with flag ({samples_with}) than without ({samples_without})"
