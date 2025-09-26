# File: scripts/train_atlas_models.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import nilearn.image as nli
from sklearn.linear_model import ElasticNetCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.lesion_analysis.features.atlas import AtlasFeatureExtractor

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# --- 1. Load Data ---
print("Loading data...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

# --- 2. Feature Engineering ---
feature_extractor = AtlasFeatureExtractor(n_rois=400, model_dir=MODELS_DIR)
feature_extractor.fit() # Creates and saves the masker if it doesn't exist

X_train_all = feature_extractor.transform(train_df)

# --- 3. Task 1: Deficit Prediction (Regression) ---
print("\n--- Training Task 1 Atlas Model (ElasticNetCV) ---")
task1_mask = (train_df.clinical_score > 0).values
X_task1 = X_train_all[task1_mask]
y_task1 = train_df.loc[task1_mask, 'clinical_score'].values

model_task1 = ElasticNetCV(cv=5, random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
model_task1.fit(X_task1, y_task1)

joblib.dump(model_task1, MODELS_DIR / "task1_atlas_model.pkl")
print(f"Task 1 model saved. Best L1 ratio: {model_task1.l1_ratio_:.2f}, Alpha: {model_task1.alpha_:.4f}")

# --- 4. Task 2: Responder Prediction (Classification) ---
print("\n--- Training Task 2 Atlas Model (XGBoost) ---")
task2_mask = train_df.treatment_assignment.notna().values
X_task2 = X_train_all[task2_mask]
y_task2 = train_df.loc[task2_mask, 'is_responder'].values

sample_weights = compute_sample_weight(class_weight='balanced', y=y_task2)

model_task2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model_task2.fit(X_task2, y_task2, sample_weight=sample_weights)

joblib.dump(model_task2, MODELS_DIR / "task2_atlas_model.pkl")
print("Task 2 model saved.")

# --- 5. Task 3: Inference Maps ---
print("\n--- Generating Atlas-Based Inference Maps ---")

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

print("\nAtlas model training script finished.")