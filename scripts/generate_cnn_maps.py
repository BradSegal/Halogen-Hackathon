import torch
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import argparse

from src.lesion_analysis.models.cnn import MultiTaskCNN
from src.lesion_analysis.models.explanability import generate_saliency_map

# --- Configuration & Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def main(args):
    RESULTS_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model and Data ---
    model = MultiTaskCNN()
    model_path = MODELS_DIR / "multitask_cnn_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    # --- Generate Deficit Map (Task 3a) ---
    print("\n--- Generating Deficit Substrate Map ---")
    # Cohort: Patients in the test set with high clinical scores.
    # Threshold is defined by the training set to avoid data leakage.
    deficit_threshold = train_df[train_df.clinical_score > 0].clinical_score.quantile(
        0.75
    )
    deficit_cohort_df = test_df[test_df.clinical_score >= deficit_threshold]
    if args.smoke_test:
        deficit_cohort_df = deficit_cohort_df.head(2)
    print(
        f"Identifying deficit map from {len(deficit_cohort_df)} high-deficit patients..."
    )

    all_deficit_maps = []
    for _, row in tqdm(
        deficit_cohort_df.iterrows(),
        total=len(deficit_cohort_df),
        desc="Deficit Map Saliency",
    ):
        img = nib.load(row["lesion_filepath"])
        img_tensor = (
            torch.from_numpy(img.get_fdata(dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        w_tensor = torch.tensor([0], dtype=torch.float32).to(
            device
        )  # W is irrelevant for severity head

        saliency_map = generate_saliency_map(
            model, img_tensor, w_tensor, target_head="severity", n_steps=50
        )
        all_deficit_maps.append(saliency_map)

    mean_deficit_map = np.mean(all_deficit_maps, axis=0)

    # --- Generate Treatment Map (Task 3b) ---
    print("\n--- Generating Treatment Response Substrate Map ---")
    # Cohort: Patients in the test set who received treatment and are PREDICTED to be responders.
    treatment_cohort_df = test_df[test_df.treatment_assignment == "Treatment"].copy()

    print(
        f"Calculating predicted CATE for {len(treatment_cohort_df)} treated patients..."
    )
    cates = []
    with torch.no_grad():
        for _, row in treatment_cohort_df.iterrows():
            img = nib.load(row["lesion_filepath"])
            img_tensor = (
                torch.from_numpy(img.get_fdata(dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # Predict outcome with treatment
            pred_treated = model(
                img_tensor, torch.tensor([1], dtype=torch.float32).to(device)
            )
            # Predict outcome without treatment (counterfactual)
            pred_control = model(
                img_tensor, torch.tensor([0], dtype=torch.float32).to(device)
            )

            # CATE = treatment benefit = outcome_treated - outcome_control
            # Positive CATE means treatment improves outcome (since higher outcome is better)
            cate = pred_treated["outcome"].item() - pred_control["outcome"].item()
            cates.append(cate)

    treatment_cohort_df["predicted_cate"] = cates
    responder_cohort_df = treatment_cohort_df[treatment_cohort_df["predicted_cate"] > 0]
    if args.smoke_test:
        responder_cohort_df = responder_cohort_df.head(2)
    print(
        f"Identifying treatment map from {len(responder_cohort_df)} predicted responders..."
    )

    all_treatment_maps = []
    for _, row in tqdm(
        responder_cohort_df.iterrows(),
        total=len(responder_cohort_df),
        desc="Treatment Map Saliency",
    ):
        img = nib.load(row["lesion_filepath"])
        img_tensor = (
            torch.from_numpy(img.get_fdata(dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        w_tensor = torch.tensor([1], dtype=torch.float32).to(
            device
        )  # Must explain outcome for treated patient

        saliency_map = generate_saliency_map(
            model, img_tensor, w_tensor, target_head="outcome", n_steps=50
        )
        all_treatment_maps.append(saliency_map)

    mean_treatment_map = (
        np.mean(all_treatment_maps, axis=0)
        if all_treatment_maps
        else np.zeros_like(mean_deficit_map)
    )

    # --- Save Maps ---
    print("\nSaving inference maps...")
    sample_affine = nib.load(test_df.iloc[0]["lesion_filepath"]).affine

    nib.save(
        nib.Nifti1Image(mean_deficit_map, sample_affine),
        RESULTS_DIR / "deficit_map_cnn.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(mean_treatment_map, sample_affine),
        RESULTS_DIR / "treatment_map_cnn.nii.gz",
    )
    print("Inference maps saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run on a small subset for testing."
    )
    args = parser.parse_args()
    main(args)
