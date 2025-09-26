import argparse
import os
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier, DummyRegressor
from tqdm import tqdm

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, default=os.getcwd())
    return parser.parse_args()

def _read_in_data(path, targets) -> np.ndarray:
    
    with zipfile.ZipFile(path, "r") as z, tempfile.TemporaryDirectory() as temp_dir:
        # Extract all files to a temporary directory
        z.extractall(temp_dir)
        # Find all NIfTI files in the extracted directory
        nii_files = [os.path.join(temp_dir, "lesions", f) for f in os.listdir(os.path.join(temp_dir, "lesions")) if f.endswith('.nii') or f.endswith('.nii.gz')]
        # Load each NIfTI file and convert to numpy array
        data_arrays = {}
        for nii_file in tqdm(nii_files, desc="Loading NIfTI files"):
            lesion_id = nii_file.split("/")[-1]
            img = nib.load(nii_file)
            data_arrays[lesion_id] = {}
            data_arrays[lesion_id]["Lesion"] = img.get_fdata()
            data_arrays[lesion_id]['Q1_clinical_score'] = targets.loc[targets['lesion_id'] == lesion_id]['Clinical score'].item()
            data_arrays[lesion_id]['Q2_treatment_assignment'] = targets.loc[targets['lesion_id'] == lesion_id]['Treatment assignment'].item()
            data_arrays[lesion_id]['Q2_outcome'] = targets.loc[targets['lesion_id'] == lesion_id]['Outcome score'].item()
        return data_arrays

def task1_fit(data: dict, model_dir: str):
    lesions = data.keys()
    X = np.array([data[k]['Lesion'] for k in lesions])
    y = np.array([data[k]['Q1_clinical_score'] for k in lesions])
    # Fit a regression model for (X, y) and save this to model_dir
    model = DummyRegressor() # Replace this with your model.
    model.fit(X, y)
    joblib.dump(model, os.path.join(model_dir, "task1_model.pkl"))


def task2_fit(data: dict, model_dir: str):
    lesions = data.keys()
    X = np.array([data[k]['Lesion'] for k in lesions]) # Lesion data
    W = np.array([data[k]['Q2_treatment_assignment'] for k in lesions]) # Treatment assignment data
    Y_baseline = np.array([data[k]['Q1_clinical_score'] for k in lesions]) # Baseline score data
    Y_outcome = np.array([data[k]['Q2_outcome'] for k in lesions]) # Outcome data

    trial_data = W != "nan"
    X = X[trial_data]
    W = W[trial_data]
    Y_outcome = Y_outcome[trial_data]
    Y_baseline = Y_baseline[trial_data]

    # Fit a prescriptive model for (X, W, Y) and save this to model_dir.
    # This model should identify the lesions which are responsive to the treatment (W) most likely to improve the outcome (Y, lower is better) given the lesion (X).
    # model = fit_model(X, W, Y_outcome, Y_baseline)
    # save_model(model, model_dir)
    model = DummyClassifier()
    model.fit(X, W == "Responsive") # You should not fit a model like this. It should infer W by which is most appropriate with respect to the outcome, Y.
    joblib.dump(model, os.path.join(model_dir, "task2_model.pkl"))

def task3(data: dict, results_path: str):
    lesions = data.keys()
    X = np.array([data[k]['Lesion'] for k in lesions]) # Lesion data
    W_trial = np.array([data[k]['Q2_treatment_assignment'] for k in lesions]) # Treatment assignment data
    Y_baseline = np.array([data[k]['Q1_clinical_score'] for k in lesions]) # Baseline score data
    Y_outcome = np.array([data[k]['Q2_outcome'] for k in lesions]) # Outcome data

    trial_data = W_trial != "nan"
    X_trial = X[trial_data]
    W_trial = W_trial[trial_data]
    Y_trial_baseline = Y_baseline[trial_data]
    Y_outcome = Y_outcome[trial_data]

    # Infer a map for the deficit-determining region.
    # deficit_map = infer_map(X, Y_baseline) # This should be numpy array of the same shape as the lesions (91, 109, 91) voxels (each 2mm^3); with 1 for the deficit-determining region and 0 for the rest.
    # save_map(deficit_map, os.path.join(results_path, "deficit_map.nii.gz")) # Save as NIfTI file
    deficit_map = np.random.randint(0, 2, (91, 109, 91)).astype(float)
    nib.save(nib.Nifti1Image(deficit_map, np.array([[2, 0, 0, -90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])), os.path.join(results_path, "deficit_map.nii.gz"))

    # Infer a map for the treatment-determining region.
    # treatment_map = infer_map(X_trial, W_trial, Y_outcome, Y_trial_baseline) # This should be numpy array of the same shape as the lesions (91, 109, 91) voxels (each 2mm^3); with 1 for the treatment-determining region and 0 for the rest.
    # save_map(treatment_map, os.path.join(results_path, "treatment_map.nii.gz")) # Save as NIfTI file
    treatment_map = np.random.randint(0, 2, (91, 109, 91)).astype(float)
    nib.save(nib.Nifti1Image(treatment_map, np.array([[2, 0, 0, -90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])), os.path.join(results_path, "treatment_map.nii.gz"))


if __name__ == "__main__":
    args = command_line_args()
    savepath = args.savepath

    data_root = Path(__file__).parent.parent.parent / 'data'

    lesionpath = data_root / "participant-accessible" / "lesions.zip"
    task_targets_path = data_root / "participant-accessible" / "tasks.csv"

    training_data = _read_in_data(lesionpath, pd.read_csv(task_targets_path))

    # Task 1: Given these sets of lesions and the arbitrary clinical score supplied; provide a model that can predict the clinical score.
    task1_fit(training_data, savepath) # Save your fitted model to path for evaluation on the external validation dataset

    # Task 2: Given these sets of lesions, treatments received (randomized) and clinical outcomes; predict responsiveness to treatment.
    task2_fit(training_data, savepath) # Save your fitted model to path for evaluation on the external validation dataset

    # Task 3: Infer two maps in MNI space (91*109*91): one inferring the spatial distribution of the deficit and the other inferring the spatial distribution of the treatment-determining region.
    task3(training_data, savepath) # Save your inferred maps to path to be sent/scored by the assessment team

    print("Completed all tasks")
