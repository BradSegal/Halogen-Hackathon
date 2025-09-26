### Ticket: TASK-01: Implement Voxel-Based Baseline Models

#### 1. Description
The primary objective of this ticket is to establish a simple, robust performance baseline for all three tasks (prediction, prescription, inference). We will create an end-to-end modeling pipeline that operates directly on voxel data. Due to the high dimensionality of the original lesion maps (~900,000 voxels), a direct approach is computationally infeasible and guaranteed to overfit.

Therefore, the core strategy will be to:
1.  Spatially downsample each lesion map to a much lower, manageable resolution.
2.  Flatten these downsampled maps into 1D feature vectors.
3.  Train simple, regularized scikit-learn models on these vectors.
4.  Generate inference maps by reshaping and upsampling the model coefficients back to the original image space.

This baseline will serve as a crucial benchmark to evaluate the effectiveness of more sophisticated approaches (like the atlas-based models) and will validate the data pipeline created in `CORE-01`.

#### 2. Justification
This task is a direct application of our core principles to establish a solid, empirical foundation for the project:

*   **Principle of Least Surprise (POLS):** This approach represents the most straightforward, "common sense" method to tackle the problem. It avoids complex feature engineering and provides a performance floor that any advanced model must significantly outperform to justify its complexity. The models chosen (`RidgeCV`, `LogisticRegressionCV`) are standard, interpretable workhorses for high-dimensional data.
*   **DRY (Don't Repeat Yourself):** This entire ticket is predicated on reusing the data artifacts (`train.csv`, `test.csv`) produced by the `CORE-01` ticket. We will not write any new data loading or cleaning logic, but instead build directly on the established foundation.
*   **Fail Fast, Fail Loudly:** As the first modeling ticket to consume the processed data, this script will act as an integration test for the entire data pipeline. Any issues with file paths, data formats, or inconsistencies in the `train.csv` file will be immediately exposed.

#### 3. Implementation Plan

**Step 1: Create Voxel Feature Engineering Module**

1.  Create a new file: `src/lesion_analysis/features/voxel.py`.
2.  Implement a function to downsample, flatten, and cache the lesion data. This caching is critical to prevent re-processing the entire dataset on every script run.

    ```python
    # File: src/lesion_analysis/features/voxel.py
    
    import nibabel as nib
    import numpy as np
    import pandas as pd
    from nilearn import image
    from pathlib import Path
    from tqdm import tqdm
    import joblib
    
    def downsample_and_flatten_lesions(
        df: pd.DataFrame,
        output_dir: Path,
        scale_factor: float = 0.25, # Downsample to 1/4 resolution
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Loads lesion files from a DataFrame, downsamples them, flattens them into
        vectors, and caches the result.
    
        Args:
            df: DataFrame containing a 'lesion_filepath' column.
            output_dir: Directory to save the cached feature array.
            scale_factor: The factor by which to downsample each dimension.
            force_recompute: If True, recomputes features even if a cached file exists.
    
        Returns:
            A numpy array of shape (n_samples, n_flattened_voxels).
        """
        output_dir.mkdir(exist_ok=True)
        # Unique cache file name based on number of samples and scale factor
        cache_filename = f"voxel_features_{len(df)}_scale_{scale_factor}.joblib"
        cache_path = output_dir / cache_filename
    
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached features from {cache_path}")
            return joblib.load(cache_path)
    
        print("Computing voxel features...")
        feature_vectors = []
        
        # Determine target shape from the first image
        first_img = nib.load(df["lesion_filepath"].iloc[0])
        original_affine = first_img.affine
        target_affine = original_affine.copy()
        target_affine[:3, :3] *= 1 / scale_factor
    
        for filepath in tqdm(df["lesion_filepath"], desc="Downsampling lesions"):
            resampled_img = image.resample_img(
                filepath, 
                target_affine=target_affine, 
                interpolation='nearest'
            )
            feature_vectors.append(resampled_img.get_fdata().flatten())
    
        features = np.array(feature_vectors)
        joblib.dump(features, cache_path)
        print(f"Saved computed features to {cache_path}")
    
        return features
    ```

**Step 2: Implement the Baseline Model Training Script**

1.  Create a new file: `scripts/train_baseline_models.py`.
2.  This script will orchestrate feature generation, model training for all tasks, and inference map generation.

    ```python
    # File: scripts/train_baseline_models.py
    
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
    task1_mask = (train_df.clinical_score > 0).values
    X_task1 = X_train_all[task1_mask]
    y_task1 = train_df.loc[task1_mask, 'clinical_score'].values
    
    model_task1 = RidgeCV(alphas=np.logspace(-3, 3, 7))
    model_task1.fit(X_task1, y_task1)
    
    # Evaluate on training set (for baseline check)
    preds = model_task1.predict(X_task1)
    rmse = np.sqrt(mean_squared_error(y_task1, preds))
    print(f"Task 1 Train RMSE: {rmse:.4f}")
    
    joblib.dump(model_task1, MODELS_DIR / "task1_baseline_model.pkl")
    print("Task 1 model saved.")
    
    # --- 4. Task 2: Responder Prediction (Classification) ---
    print("\n--- Training Task 2 Model ---")
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
    print(f"Task 2 Train Balanced Accuracy: {bacc:.4f}")
    
    joblib.dump(model_task2, MODELS_DIR / "task2_baseline_model.pkl")
    print("Task 2 model saved.")
    
    # --- 5. Task 3: Inference Maps ---
    print("\n--- Generating Inference Maps ---")
    
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
    
    print("\nBaseline script finished.")
    ```

#### 4. Acceptance Criteria
*   [ ] The file `src/lesion_analysis/features/voxel.py` exists and contains the `downsample_and_flatten_lesions` function.
*   [ ] The script `scripts/train_baseline_models.py` runs to completion without errors.
*   [ ] The script generates and saves the following output files:
    *   `models/task1_baseline_model.pkl`
    *   `models/task2_baseline_model.pkl`
    *   `results/deficit_map_baseline.nii.gz`
    *   `results/treatment_map_baseline.nii.gz`
*   [ ] The script prints the training set RMSE for Task 1 and Balanced Accuracy for Task 2 to the console.
*   [ ] A cached feature file (e.g., `voxel_features_*.joblib`) is created in `data/processed/features_cache/`.

#### 5. Testing Requirements
*   **Unit Tests:**
    *   Create `tests/features/test_voxel.py`.
    *   Write a test for `downsample_and_flatten_lesions`.
        *   Use `pytest.fixture` to create a temporary directory with a few dummy 3D NIfTI files and a mock DataFrame.
        *   Call the function and assert that the returned numpy array has the correct shape: `(n_samples, d1*d2*d3)` where `d` are the downsampled dimensions.
        *   Assert that the cache file is created.
        *   Call the function a second time and confirm (e.g., by checking log output with `capsys`) that it loads from the cache instead of re-computing.
*   **Integration Tests:**
    *   Create `tests/scripts/test_train_baseline_models.py`.
    *   Write a test that runs the `scripts/train_baseline_models.py` script as a subprocess or by importing its main function.
    *   Use a fixture to create a minimal, valid `train.csv` file (e.g., 20 rows) pointing to real lesion files from the dataset.
    *   Assert that all four output files (`.pkl` and `.nii.gz`) are created in the expected locations.
    *   Use `nibabel` to load the generated `.nii.gz` maps and assert their dimensions match the original lesion map dimensions (`91, 109, 91`).

#### 6. Definition of Done
*   [ ] All Acceptance Criteria are met.
*   [ ] All required unit and integration tests are written and pass successfully.
*   [ ] The code has been formatted (`black`), linted (`ruff`), and type-checked (`mypy`) successfully.
*   [ ] The project `README.md` is updated to include instructions for running the baseline model training script.
*   [ ] The code has been peer-reviewed and approved.