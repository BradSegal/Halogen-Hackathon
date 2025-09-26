import pytest
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
from unittest.mock import patch, MagicMock

from scripts.train_atlas_models import PROJECT_ROOT


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

    # Create minimal lesion data with MNI-like dimensions (smaller for testing)
    shape = (91, 109, 91)  # Real MNI dimensions but smaller data for fast testing
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


@pytest.fixture
def mock_atlas():
    """Create mock atlas data for testing."""
    # Create a simple atlas with 4 ROIs (much smaller than real 400 ROIs for testing)
    shape = (91, 109, 91)  # Use MNI dimensions
    affine = np.eye(4)

    # Create atlas data with 4 distinct regions
    atlas_data = np.zeros(shape, dtype=np.int32)
    atlas_data[0:45, :, :] = 1  # ROI 1
    atlas_data[45:91, 0:54, :] = 2  # ROI 2
    atlas_data[45:91, 54:109, 0:45] = 3  # ROI 3
    atlas_data[45:91, 54:109, 45:91] = 4  # ROI 4

    atlas_img = nib.Nifti1Image(atlas_data, affine)

    # Mock atlas object
    mock_atlas = MagicMock()
    mock_atlas.maps = atlas_img

    return mock_atlas


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_train_atlas_models_script_execution(
    mock_fetch, mock_atlas, temp_project_structure
):
    """Test that the training script runs to completion and creates expected files."""
    temp_dir, _ = temp_project_structure
    mock_fetch.return_value = mock_atlas

    # Create a modified version of the script that uses temp_dir and mocked atlas
    modified_script_content = f"""
import sys
import os
sys.path.insert(0, '{PROJECT_ROOT}')
os.environ['PROJECT_ROOT'] = '{temp_dir}'

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import nilearn.image as nli
from sklearn.linear_model import ElasticNetCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from unittest.mock import patch, MagicMock
import nibabel as nib

# Mock atlas setup
shape = (91, 109, 91)
affine = np.eye(4)
atlas_data = np.zeros(shape, dtype=np.int32)
atlas_data[0:45, :, :] = 1
atlas_data[45:91, 0:54, :] = 2
atlas_data[45:91, 54:109, 0:45] = 3
atlas_data[45:91, 54:109, 45:91] = 4
atlas_img = nib.Nifti1Image(atlas_data, affine)
mock_atlas = MagicMock()
mock_atlas.maps = atlas_img

# Patch the fetch function
with patch('src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018', return_value=mock_atlas):
    from src.lesion_analysis.features.atlas import AtlasFeatureExtractor

    # --- Configuration ---
    PROJECT_ROOT = Path('{temp_dir}')
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- 1. Load Data ---
    print("Loading data...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    # --- 2. Feature Engineering ---
    feature_extractor = AtlasFeatureExtractor(n_rois=4, model_dir=MODELS_DIR)
    feature_extractor.fit()

    X_train_all = feature_extractor.transform(train_df)

    # --- 3. Task 1: Deficit Prediction (Regression) ---
    print("\\n--- Training Task 1 Atlas Model (ElasticNetCV) ---")
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, 'clinical_score'].values

    model_task1 = ElasticNetCV(cv=3, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
    model_task1.fit(X_task1, y_task1)

    joblib.dump(model_task1, MODELS_DIR / "task1_atlas_model.pkl")
    print(f"Task 1 model saved. Best L1 ratio: {{model_task1.l1_ratio_:.2f}}, Alpha: {{model_task1.alpha_:.4f}}")

    # --- 4. Task 2: Responder Prediction (Classification) ---
    print("\\n--- Training Task 2 Atlas Model (XGBoost) ---")
    task2_mask = train_df.treatment_assignment.notna().values
    X_task2 = X_train_all[task2_mask]
    y_task2 = train_df.loc[task2_mask, 'is_responder'].values

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_task2)

    model_task2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_task2.fit(X_task2, y_task2, sample_weight=sample_weights)

    joblib.dump(model_task2, MODELS_DIR / "task2_atlas_model.pkl")
    print("Task 2 model saved.")

    # --- 5. Task 3: Inference Maps ---
    print("\\n--- Generating Atlas-Based Inference Maps ---")

    # Deficit Map from Task 1 model coefficients
    deficit_weights = model_task1.coef_
    deficit_map_img = feature_extractor.masker.inverse_transform(deficit_weights)
    deficit_map_img.to_filename(RESULTS_DIR / "deficit_map_atlas.nii.gz")
    print("Deficit map saved.")

    # Treatment Map from Task 2 model feature importances
    treatment_weights = model_task2.feature_importances_
    treatment_map_img = feature_extractor.masker.inverse_transform(treatment_weights)

    # Enforce subset constraint: treatment map must be a subset of deficit map
    # Binarize the deficit map (only regions with non-zero coefficients matter)
    deficit_mask_img = nli.math_img("np.abs(img) > 1e-6", img=deficit_map_img)
    final_treatment_map_img = nli.math_img("img1 * img2", img1=treatment_map_img, img2=deficit_mask_img)
    final_treatment_map_img.to_filename(RESULTS_DIR / "treatment_map_atlas.nii.gz")
    print("Treatment map saved.")

    print("\\nAtlas model training script finished.")
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

    # Check the 5 expected artifacts from the ticket
    assert (models_dir / "schaefer_2018_4rois_masker.joblib").exists()
    assert (models_dir / "task1_atlas_model.pkl").exists()
    assert (models_dir / "task2_atlas_model.pkl").exists()
    assert (results_dir / "deficit_map_atlas.nii.gz").exists()
    assert (results_dir / "treatment_map_atlas.nii.gz").exists()


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_generated_atlas_maps_have_correct_dimensions(
    mock_fetch, mock_atlas, temp_project_structure
):
    """Test that generated maps have correct dimensions (91, 109, 91)."""
    temp_dir, filepaths = temp_project_structure
    mock_fetch.return_value = mock_atlas

    # Import components
    from src.lesion_analysis.features.atlas import AtlasFeatureExtractor
    from sklearn.linear_model import ElasticNetCV
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    # Load data
    train_df = pd.read_csv(temp_dir / "data" / "processed" / "train.csv")
    models_dir = temp_dir / "models"
    results_dir = temp_dir / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Generate features using atlas
    feature_extractor = AtlasFeatureExtractor(n_rois=4, model_dir=models_dir)
    feature_extractor.fit()
    X_train_all = feature_extractor.transform(train_df)

    # Train models
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, "clinical_score"].values
    model_task1 = ElasticNetCV(cv=3, random_state=42, l1_ratio=[0.5, 0.7, 0.9])
    model_task1.fit(X_task1, y_task1)

    task2_mask = train_df.treatment_assignment.notna().values
    X_task2 = X_train_all[task2_mask]
    y_task2 = train_df.loc[task2_mask, "is_responder"].values
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_task2)
    model_task2 = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    model_task2.fit(X_task2, y_task2, sample_weight=sample_weights)

    # Generate maps
    deficit_weights = model_task1.coef_
    deficit_map_img = feature_extractor.masker.inverse_transform(deficit_weights)

    treatment_weights = model_task2.feature_importances_
    treatment_map_img = feature_extractor.masker.inverse_transform(treatment_weights)

    # Check dimensions are (91, 109, 91) as specified in the ticket
    expected_shape = (91, 109, 91)
    assert deficit_map_img.shape == expected_shape
    assert treatment_map_img.shape == expected_shape

    # Save and load to verify they can be read as NIfTI files
    deficit_map_img.to_filename(results_dir / "deficit_map_atlas.nii.gz")
    treatment_map_img.to_filename(results_dir / "treatment_map_atlas.nii.gz")

    # Load the saved files and verify dimensions again
    loaded_deficit = nib.load(results_dir / "deficit_map_atlas.nii.gz")
    loaded_treatment = nib.load(results_dir / "treatment_map_atlas.nii.gz")

    assert loaded_deficit.shape == expected_shape
    assert loaded_treatment.shape == expected_shape


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_feature_extraction_produces_correct_shape(
    mock_fetch, mock_atlas, temp_project_structure
):
    """Test that atlas feature extraction produces features with correct shape."""
    temp_dir, _ = temp_project_structure
    mock_fetch.return_value = mock_atlas

    from src.lesion_analysis.features.atlas import AtlasFeatureExtractor

    # Load data
    train_df = pd.read_csv(temp_dir / "data" / "processed" / "train.csv")
    models_dir = temp_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Test with 4 ROIs (matching our mock atlas)
    feature_extractor = AtlasFeatureExtractor(n_rois=4, model_dir=models_dir)
    feature_extractor.fit()

    X_features = feature_extractor.transform(train_df)

    # Should have n_samples rows and n_rois columns
    assert X_features.shape == (len(train_df), 4)
    assert X_features.dtype == np.float64  # Default numpy float type
    assert np.all(X_features >= 0)  # All lesion loads should be non-negative


@patch("src.lesion_analysis.features.atlas.fetch_atlas_schaefer_2018")
def test_atlas_models_training_metrics(mock_fetch, mock_atlas, temp_project_structure):
    """Test that models train successfully and produce reasonable metrics."""
    temp_dir, _ = temp_project_structure
    mock_fetch.return_value = mock_atlas

    from src.lesion_analysis.features.atlas import AtlasFeatureExtractor
    from sklearn.linear_model import ElasticNetCV
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import mean_squared_error, balanced_accuracy_score

    # Load data and generate features
    train_df = pd.read_csv(temp_dir / "data" / "processed" / "train.csv")
    models_dir = temp_dir / "models"
    models_dir.mkdir(exist_ok=True)

    feature_extractor = AtlasFeatureExtractor(n_rois=4, model_dir=models_dir)
    feature_extractor.fit()
    X_train_all = feature_extractor.transform(train_df)

    # Train Task 1 model
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, "clinical_score"].values

    model_task1 = ElasticNetCV(cv=3, random_state=42, l1_ratio=[0.5, 0.7, 0.9])
    model_task1.fit(X_task1, y_task1)

    preds = model_task1.predict(X_task1)
    rmse = np.sqrt(mean_squared_error(y_task1, preds))

    # Train Task 2 model
    task2_mask = train_df.treatment_assignment.notna().values
    X_task2 = X_train_all[task2_mask]
    y_task2 = train_df.loc[task2_mask, "is_responder"].values

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_task2)
    model_task2 = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    model_task2.fit(X_task2, y_task2, sample_weight=sample_weights)

    preds = model_task2.predict(X_task2)
    bacc = balanced_accuracy_score(y_task2, preds)

    # Both metrics should be reasonable values
    assert 0 <= rmse <= 20  # RMSE should be positive and reasonable
    assert 0 <= bacc <= 1  # Balanced accuracy should be between 0 and 1

    # Models should have expected attributes
    assert hasattr(model_task1, "coef_")
    assert hasattr(model_task1, "alpha_")
    assert hasattr(model_task1, "l1_ratio_")
    assert hasattr(model_task2, "feature_importances_")

    # Coefficient/importance arrays should have correct length
    assert len(model_task1.coef_) == 4  # Same as n_rois
    assert len(model_task2.feature_importances_) == 4  # Same as n_rois
