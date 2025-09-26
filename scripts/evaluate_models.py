# File: scripts/evaluate_models.py

import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
import sys
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
import logging

# --- Setup ---
# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions
from src.lesion_analysis.features.atlas import AtlasFeatureExtractor
from src.lesion_analysis.models.cnn import Simple3DCNN
from src.lesion_analysis.models.torch_loader import LesionDataset
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_CACHE_DIR = PROCESSED_DATA_DIR / "features_cache"

RESULTS_DIR.mkdir(exist_ok=True)

# --- Helper Functions for each model type ---


def evaluate_baseline_models(test_df: pd.DataFrame) -> dict:
    """Evaluates the baseline models. Returns a dictionary of results."""
    logger.info("--- Evaluating Baseline Models ---")
    results = {}
    try:
        # Task 1 (Regression)
        model_path = MODELS_DIR / "task1_baseline_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_b1 = joblib.load(model_path)
        df_t1 = test_df[test_df.clinical_score > 0]
        X_t1 = downsample_and_flatten_lesions(df_t1, FEATURES_CACHE_DIR)
        preds = model_b1.predict(X_t1)
        rmse = np.sqrt(mean_squared_error(df_t1.clinical_score, preds))
        results["Baseline_Task1_RMSE"] = rmse
        logger.info(f"  - Task 1 RMSE: {rmse:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"  - Could not evaluate Baseline Task 1 model: {e}")

    try:
        # Task 2 (Classification)
        model_path = MODELS_DIR / "task2_baseline_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_b2 = joblib.load(model_path)
        df_t2 = test_df[test_df.treatment_assignment.notna()]
        X_t2 = downsample_and_flatten_lesions(df_t2, FEATURES_CACHE_DIR)
        preds = model_b2.predict(X_t2)
        bacc = balanced_accuracy_score(df_t2.is_responder.astype(bool), preds)
        results["Baseline_Task2_BACC"] = bacc
        logger.info(f"  - Task 2 Balanced Accuracy: {bacc:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"  - Could not evaluate Baseline Task 2 model: {e}")

    return results


def evaluate_atlas_models(test_df: pd.DataFrame) -> dict:
    """Evaluates the atlas-based models. Returns a dictionary of results."""
    logger.info("--- Evaluating Atlas Models ---")
    results = {}
    try:
        feature_extractor_atlas = AtlasFeatureExtractor(
            n_rois=400, model_dir=MODELS_DIR
        )

        # Task 1 (Regression)
        model_path = MODELS_DIR / "task1_atlas_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_a1 = joblib.load(model_path)
        df_t1 = test_df[test_df.clinical_score > 0]
        X_t1_atlas = feature_extractor_atlas.transform(df_t1)
        preds = model_a1.predict(X_t1_atlas)
        rmse = np.sqrt(mean_squared_error(df_t1.clinical_score, preds))
        results["Atlas_Task1_RMSE"] = rmse
        logger.info(f"  - Task 1 RMSE: {rmse:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"  - Could not evaluate Atlas Task 1 model: {e}")

    try:
        feature_extractor_atlas = AtlasFeatureExtractor(
            n_rois=400, model_dir=MODELS_DIR
        )

        # Task 2 (Classification)
        model_path = MODELS_DIR / "task2_atlas_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_a2 = joblib.load(model_path)
        df_t2 = test_df[test_df.treatment_assignment.notna()]
        X_t2_atlas = feature_extractor_atlas.transform(df_t2)
        preds = model_a2.predict(X_t2_atlas)
        bacc = balanced_accuracy_score(df_t2.is_responder.astype(bool), preds)
        results["Atlas_Task2_BACC"] = bacc
        logger.info(f"  - Task 2 Balanced Accuracy: {bacc:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"  - Could not evaluate Atlas Task 2 model: {e}")

    return results


def evaluate_cnn_models(test_df: pd.DataFrame) -> dict:
    """Evaluates the CNN models. Returns a dictionary of results."""
    logger.info("--- Evaluating CNN Models ---")
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Task 1 (Regression)
        model_path = MODELS_DIR / "task1_cnn_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_c1 = Simple3DCNN().to(device)
        model_c1.load_state_dict(torch.load(model_path, map_location=device))
        model_c1.eval()

        df_t1 = test_df[test_df.clinical_score > 0]
        ds_c1 = LesionDataset(df_t1, "clinical_score")
        loader_c1 = DataLoader(ds_c1, batch_size=8, shuffle=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, targets in loader_c1:
                imgs = imgs.to(device)
                preds = model_c1(imgs)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())

        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        results["CNN_Task1_RMSE"] = rmse
        logger.info(f"  - Task 1 RMSE: {rmse:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"  - Could not evaluate CNN Task 1 model: {e}")

    try:
        # Task 2 (Classification)
        model_path = MODELS_DIR / "task2_cnn_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_c2 = Simple3DCNN().to(device)
        model_c2.load_state_dict(torch.load(model_path, map_location=device))
        model_c2.eval()

        df_t2 = test_df[test_df.treatment_assignment.notna()]
        ds_c2 = LesionDataset(df_t2, "is_responder")
        loader_c2 = DataLoader(ds_c2, batch_size=8, shuffle=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, targets in loader_c2:
                imgs = imgs.to(device)
                logits = model_c2(imgs)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())

        bacc = balanced_accuracy_score(all_targets, all_preds)
        results["CNN_Task2_BACC"] = bacc
        logger.info(f"  - Task 2 Balanced Accuracy: {bacc:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"  - Could not evaluate CNN Task 2 model: {e}")

    return results


def main():
    """Main function to run the evaluation pipeline."""
    try:
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    except FileNotFoundError:
        logger.error(
            f"FATAL: test.csv not found at {PROCESSED_DATA_DIR}. Please run scripts/prepare_data.py first."
        )
        return

    all_results = {}
    all_results.update(evaluate_baseline_models(test_df))
    all_results.update(evaluate_atlas_models(test_df))
    all_results.update(evaluate_cnn_models(test_df))

    if not all_results:
        logger.error("No models were found to evaluate. Please run training scripts.")
        return

    # --- Report Results ---
    logger.info("\n--- Final Model Performance on Test Set ---")
    results_df = pd.DataFrame.from_dict(all_results, orient="index", columns=["Score"])
    results_df.index.name = "Model_Metric"
    print(results_df.to_string())

    output_path = RESULTS_DIR / "evaluation_report.csv"
    results_df.to_csv(output_path)
    logger.info(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
