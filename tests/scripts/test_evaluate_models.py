"""Integration tests for the evaluation pipeline."""

import sys
from pathlib import Path

# Setup paths and add to sys.path BEFORE other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pytest  # noqa: E402
import subprocess  # noqa: E402
import pandas as pd  # noqa: E402
import shutil  # noqa: E402
import joblib  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.dummy import DummyRegressor, DummyClassifier  # noqa: E402
from src.lesion_analysis.models.cnn import Simple3DCNN  # noqa: E402

# Configuration paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVALUATION_SCRIPT = SCRIPTS_DIR / "evaluate_models.py"
EVALUATION_REPORT = RESULTS_DIR / "evaluation_report.csv"


@pytest.fixture
def clean_models_dir():
    """Fixture to backup and restore models directory."""
    # Create backup directory
    backup_dir = MODELS_DIR.parent / "models_backup"

    # Backup existing models if directory exists
    if MODELS_DIR.exists():
        shutil.copytree(MODELS_DIR, backup_dir, dirs_exist_ok=True)

    yield

    # Restore from backup
    if backup_dir.exists():
        shutil.rmtree(MODELS_DIR, ignore_errors=True)
        shutil.copytree(backup_dir, MODELS_DIR)
        shutil.rmtree(backup_dir)


@pytest.fixture
def clean_report():
    """Fixture to clean up evaluation report after test."""
    yield
    # Clean up report file if it exists
    if EVALUATION_REPORT.exists():
        EVALUATION_REPORT.unlink()


def create_dummy_baseline_models():
    """Create dummy baseline models for testing."""
    MODELS_DIR.mkdir(exist_ok=True)

    # Create dummy regression model for Task 1
    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit([[0]], [0])  # Fit with minimal data
    joblib.dump(dummy_reg, MODELS_DIR / "task1_baseline_model.pkl")

    # Create dummy classification model for Task 2
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit([[0]], [0])  # Fit with minimal data
    joblib.dump(dummy_clf, MODELS_DIR / "task2_baseline_model.pkl")


def create_dummy_atlas_models():
    """Create dummy atlas models for testing."""
    MODELS_DIR.mkdir(exist_ok=True)

    # Ensure the atlas masker exists (required by AtlasFeatureExtractor)
    atlas_path = MODELS_DIR / "schaefer_2018_400rois_masker.joblib"
    if not atlas_path.exists():
        # Create a minimal dummy masker object
        class DummyMasker:
            def transform(self, X):
                # Return dummy features of shape (n_samples, 400)
                return np.random.randn(len(X), 400)

        joblib.dump(DummyMasker(), atlas_path)

    # Create dummy regression model for Task 1
    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit(np.random.randn(10, 400), np.random.randn(10))
    joblib.dump(dummy_reg, MODELS_DIR / "task1_atlas_model.pkl")

    # Create dummy classification model for Task 2
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(np.random.randn(10, 400), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    joblib.dump(dummy_clf, MODELS_DIR / "task2_atlas_model.pkl")


def create_dummy_cnn_models():
    """Create dummy CNN models for testing."""
    MODELS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create and save dummy CNN for Task 1
    model1 = Simple3DCNN().to(device)
    torch.save(model1.state_dict(), MODELS_DIR / "task1_cnn_model.pt")

    # Create and save dummy CNN for Task 2
    model2 = Simple3DCNN().to(device)
    torch.save(model2.state_dict(), MODELS_DIR / "task2_cnn_model.pt")


def run_evaluation_script():
    """Run the evaluation script and return the result."""
    result = subprocess.run(
        [sys.executable, str(EVALUATION_SCRIPT)], capture_output=True, text=True
    )
    return result


class TestEvaluationPipeline:
    """Test cases for the resilient evaluation pipeline."""

    @pytest.mark.integration
    def test_all_models_present(self, clean_models_dir, clean_report):
        """Test Case 1: All models are present and evaluation runs successfully."""
        # Create all dummy models
        create_dummy_baseline_models()
        create_dummy_atlas_models()
        create_dummy_cnn_models()

        # Run evaluation
        result = run_evaluation_script()

        # Check that script ran successfully
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

        # Check that report was created
        assert EVALUATION_REPORT.exists(), "Evaluation report was not created"

        # Check report contents
        df = pd.read_csv(EVALUATION_REPORT, index_col=0)

        # Should have results for all model types (baseline, atlas, CNN) and both tasks
        expected_metrics = [
            "Baseline_Task1_RMSE",
            "Baseline_Task2_BACC",
            "Atlas_Task1_RMSE",
            "Atlas_Task2_BACC",
            "CNN_Task1_RMSE",
            "CNN_Task2_BACC",
        ]

        for metric in expected_metrics:
            assert metric in df.index, f"Missing metric: {metric}"

    @pytest.mark.integration
    def test_some_models_missing(self, clean_models_dir, clean_report):
        """Test Case 2: Some models are missing, script handles gracefully."""
        # Create only baseline and atlas models, skip CNN
        create_dummy_baseline_models()
        create_dummy_atlas_models()

        # Run evaluation
        result = run_evaluation_script()

        # Check that script ran successfully despite missing models
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

        # Check that warnings were printed for missing CNN models
        assert "Could not evaluate CNN Task 1 model" in result.stderr
        assert "Could not evaluate CNN Task 2 model" in result.stderr

        # Check that report was created
        assert EVALUATION_REPORT.exists(), "Evaluation report was not created"

        # Check report contents
        df = pd.read_csv(EVALUATION_REPORT, index_col=0)

        # Should have results for baseline and atlas, but not CNN
        assert "Baseline_Task1_RMSE" in df.index
        assert "Baseline_Task2_BACC" in df.index
        assert "Atlas_Task1_RMSE" in df.index
        assert "Atlas_Task2_BACC" in df.index
        assert "CNN_Task1_RMSE" not in df.index
        assert "CNN_Task2_BACC" not in df.index

    @pytest.mark.integration
    def test_no_models_present(self, clean_models_dir, clean_report):
        """Test Case 3: No models are present, script exits gracefully."""
        # Ensure models directory is empty
        if MODELS_DIR.exists():
            shutil.rmtree(MODELS_DIR)
        MODELS_DIR.mkdir(exist_ok=True)

        # Run evaluation
        result = run_evaluation_script()

        # Check that script ran (may have exit code 0 even with error message)
        assert result.returncode == 0, f"Script failed unexpectedly: {result.stderr}"

        # Check that appropriate error message was printed
        assert (
            "No models were found to evaluate" in result.stderr
            or "No models were found to evaluate" in result.stdout
        )

        # Check that no report was created
        assert (
            not EVALUATION_REPORT.exists()
        ), "Report should not be created when no models exist"

    @pytest.mark.integration
    def test_partial_model_failures(self, clean_models_dir, clean_report):
        """Test that evaluation continues even if some models exist but fail during evaluation."""
        # Create baseline models only for Task 1
        MODELS_DIR.mkdir(exist_ok=True)

        dummy_reg = DummyRegressor(strategy="mean")
        dummy_reg.fit([[0]], [0])
        joblib.dump(dummy_reg, MODELS_DIR / "task1_baseline_model.pkl")

        # Run evaluation
        result = run_evaluation_script()

        # Check that script ran successfully
        assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"

        # Check that warnings were printed for missing models
        assert "Could not evaluate" in result.stderr

        # Check if report was created (should be created if at least one model succeeded)
        if EVALUATION_REPORT.exists():
            df = pd.read_csv(EVALUATION_REPORT, index_col=0)
            assert "Baseline_Task1_RMSE" in df.index


@pytest.mark.integration
def test_evaluation_script_exists():
    """Test that the evaluation script file exists and is executable."""
    assert (
        EVALUATION_SCRIPT.exists()
    ), f"Evaluation script not found at {EVALUATION_SCRIPT}"

    # Check that the script has the main guard
    with open(EVALUATION_SCRIPT, "r") as f:
        content = f.read()
        assert 'if __name__ == "__main__":' in content
        assert "def main():" in content


@pytest.mark.integration
def test_data_prerequisites():
    """Test that required data files exist for evaluation."""
    test_csv = DATA_DIR / "test.csv"
    assert (
        test_csv.exists()
    ), f"Test data not found at {test_csv}. Run scripts/prepare_data.py first."

    # Verify the test CSV has required columns
    df = pd.read_csv(test_csv)
    required_cols = [
        "clinical_score",
        "treatment_assignment",
        "is_responder",
        "lesion_path",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
