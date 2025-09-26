import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn import image

from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, balanced_accuracy_score

from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions

# --- Parse Arguments ---
parser = argparse.ArgumentParser(
    description="Train baseline models for lesion analysis"
)
parser.add_argument(
    "--train-on-all-data",
    action="store_true",
    help="If set, train Task 1 on all data, not just score > 0.",
)
args = parser.parse_args()

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
print("\n--- Training Task 1 Model ---")
if args.train_on_all_data:
    print("Training on ALL data (scores >= 0).")
    X_task1 = X_train_all
    y_task1 = train_df["clinical_score"].values
else:
    print("Training on non-zero score data only (scores > 0).")
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, "clinical_score"].values

model_task1 = RidgeCV(alphas=np.logspace(-3, 3, 7))
model_task1.fit(X_task1, y_task1)

# Evaluate on training set (for baseline check)
preds = model_task1.predict(X_task1)
rmse = np.sqrt(mean_squared_error(y_task1, preds))
print(f"Task 1 Train RMSE: {rmse:.4f}")
print(f"Training samples used: {len(X_task1)}")

joblib.dump(model_task1, MODELS_DIR / "task1_baseline_model.pkl")
print("Task 1 model saved.")

# --- 4. Task 2: Responder Prediction (Classification) ---
print("\n--- Training Task 2 Model ---")
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

# Evaluate on training set
preds = model_task2.predict(X_task2)
bacc = balanced_accuracy_score(y_task2, preds)
print(f"Task 2 Train Balanced Accuracy: {bacc:.4f}")

joblib.dump(model_task2, MODELS_DIR / "task2_baseline_model.pkl")
print("Task 2 model saved.")

# --- 5. Task 3: Inference Maps ---
print("\n--- Generating Inference Maps ---")

# Get shape info from a sample image
sample_img = nib.load(train_df["lesion_filepath"].iloc[0])
resampled_img = image.resample_img(
    sample_img, target_affine=sample_img.affine * 4, interpolation="nearest"
)
downsampled_shape = resampled_img.shape

# Deficit Map (from Ridge model)
deficit_coefs = model_task1.coef_
deficit_map_downsampled = deficit_coefs.reshape(downsampled_shape)
deficit_map_upsampled_img = image.resample_to_img(
    source_img=nib.Nifti1Image(deficit_map_downsampled, resampled_img.affine),
    target_img=sample_img,
    interpolation="continuous",
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
    interpolation="continuous",
)

# Enforce subset constraint
deficit_mask_img = image.math_img("img > 0", img=deficit_map_upsampled_img)
final_treatment_map_img = image.math_img(
    "img1 * img2", img1=treatment_map_upsampled_img, img2=deficit_mask_img
)
final_treatment_map_img.to_filename(RESULTS_DIR / "treatment_map_baseline.nii.gz")
print("Treatment map saved.")

print("\nBaseline script finished.")
