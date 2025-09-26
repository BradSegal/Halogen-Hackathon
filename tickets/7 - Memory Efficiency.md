I have reviewed the codebase with a focus on the memory consumption of the atlas script. An Out-of-Memory (OOM) error during the `inverse_transform` step is a classic `nilearn` issue when dealing with high-resolution atlases and large coefficient/importance vectors.

Let's break down the problem with precision.

### **1. Diagnosis of the Memory Overflow**

*   **Symptom:** The script `scripts/train_atlas_models.py` crashes with a memory error during Task 3, specifically at the `inverse_transform` call.
*   **Location:**
    ```python
    # In scripts/train_atlas_models.py
    deficit_map_img = feature_extractor.masker.inverse_transform(deficit_weights) # <--- THIS LINE
    ```
*   **Root Cause Analysis:** The `NiftiLabelsMasker.inverse_transform` method works by creating a new 3D volume in memory with the full dimensions of the atlas (`91, 109, 91`). It then iterates through each of the 400 ROIs defined in the atlas map. For each ROI, it finds all the voxels belonging to that region and assigns them the corresponding weight from the input vector (`deficit_weights`).

    The problem is that this process can be extremely memory-intensive. It creates and holds large intermediate arrays in memory. If the atlas is high-resolution (like the Schaefer atlas we are using), the internal arrays `nilearn` uses to perform this mapping can become very large, leading to an OOM error, especially in a memory-constrained environment. The `nilearn_cache` helps with the `transform` step but does not mitigate the memory pressure of `inverse_transform`.

### **2. The Solution: A Memory-Efficient `inverse_transform`**

We cannot change the fundamental logic of `nilearn`. Therefore, the solution is to **not use the built-in `inverse_transform` method**. We must implement our own memory-efficient version.

The logic is straightforward:
1.  Load the atlas map NIfTI file once.
2.  Get the 3D data array from the atlas map.
3.  Create an empty 3D array of the same shape, initialized to zeros, to hold our output map.
4.  Iterate from `i = 1` to `n_rois`. In each iteration:
    *   Find the coordinates of all voxels where the atlas array is equal to `i`.
    *   Assign the `i-1`-th weight from our model's coefficient vector to these coordinates in our output map.
5.  Save the final output map as a new NIfTI image.

This approach processes one ROI at a time, using boolean masking, which is significantly more memory-efficient than `nilearn`'s internal method for this specific task.

### **3. Synthesized Plan: The Implementation Ticket**

We will create a ticket to refactor the atlas feature extractor and the training script to use this robust, memory-efficient method.

---

### Ticket: REFACTOR-06: Implement Memory-Efficient Inverse Transform for Atlas Maps

#### 1. Description
The `scripts/train_atlas_models.py` script is crashing with Out-of-Memory (OOM) errors during Task 3 when calling `NiftiLabelsMasker.inverse_transform`. This is a known issue in `nilearn` where the default implementation is not memory-efficient for high-resolution atlases.

This ticket resolves the OOM error by replacing the problematic `nilearn` function call with a custom, memory-efficient implementation. We will add a new static method to the `AtlasFeatureExtractor` class that performs the inverse transform using a simple, iterative numpy-based approach. The training script will be updated to call this new, robust method to generate the final inference maps.

#### 2. Justification
*   **Fail Fast, Fail Loudly:** An OOM crash is an ungraceful failure. This fix ensures the script completes its primary function without crashing due to predictable memory constraints, making the pipeline robust.
*   **DRY (Don't Repeat Yourself):** By implementing the logic as a reusable static method within the `AtlasFeatureExtractor` class, we create a single, well-defined function for this operation that can be used by any script that needs to convert an atlas-based feature vector back into a 3D map.
*   **Principle of Least Surprise (POLS):** A script designed to generate output files should not crash during the final file-generation step. This fix ensures the script behaves as expected and successfully produces all its intended artifacts.

#### 3. Implementation Plan

**Step 1: Add a Memory-Efficient `inverse_transform` to `AtlasFeatureExtractor`**

1.  Open `src/lesion_analysis/features/atlas.py`.
2.  Add `nibabel` to the imports.
3.  Add a new static method to the `AtlasFeatureExtractor` class as shown below.

    ```python
    # File: src/lesion_analysis/features/atlas.py
    
    import joblib
    import numpy as np
    import pandas as pd
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker
    from pathlib import Path
    import nibabel as nib
    
    class AtlasFeatureExtractor:
        # ... (__init__, fit, transform methods are unchanged) ...
    
        @staticmethod
        def memory_efficient_inverse_transform(
            weights: np.ndarray, 
            atlas_img: nib.Nifti1Image
        ) -> nib.Nifti1Image:
            """
            A memory-efficient implementation of nilearn's inverse_transform.
            
            Args:
                weights: A 1D numpy array of weights (e.g., model coefficients).
                atlas_img: The NIfTI image object of the atlas itself.
                
            Returns:
                A new NIfTI image with the weights projected back into 3D space.
            """
            if weights.ndim != 1:
                raise ValueError("Weights must be a 1D array.")
            
            atlas_data = atlas_img.get_fdata()
            output_map = np.zeros_like(atlas_data, dtype=np.float32)
            
            # The number of ROIs should match the length of the weights vector.
            # Atlas labels are typically 1-based, so they go from 1 to n_rois.
            num_rois = len(weights)
            
            for i in range(num_rois):
                roi_label = i + 1  # Atlas labels are 1-based
                weight = weights[i]
                output_map[atlas_data == roi_label] = weight
                
            return nib.Nifti1Image(output_map, atlas_img.affine, atlas_img.header)

    ```
**Step 2: Update the `fit` method to store the atlas image**

1.  The new static method needs the atlas NIfTI object. We'll modify the `fit` method to load it as an instance attribute.

    ```python
    # In src/lesion_analysis/features/atlas.py
    
    class AtlasFeatureExtractor:
        def __init__(self, n_rois: int = 400, model_dir: Path = Path("models")):
            # ... (unchanged) ...
            self.atlas_img: nib.Nifti1Image = None # Add this attribute
    
        def fit(self):
            # ...
            print("Fetching atlas and creating masker...")
            atlas = fetch_atlas_schaefer_2018(n_rois=self.n_rois, resolution_mm=2)
            self.atlas_img = nib.load(atlas.maps) # Add this line to load and store the image
            # ... (masker creation is unchanged) ...
    ```

**Step 3: Update `scripts/train_atlas_models.py` to use the new method**

1.  Open the script.
2.  Modify the "Inference Maps" section to call our new static method instead of the `nilearn` one.

    ```python
    # In scripts/train_atlas_models.py
    
    # ... (after model_task2 is trained) ...
    
    # --- 5. Task 3: Inference Maps ---
    print("\n--- Generating Atlas-Based Inference Maps ---")
    
    # Deficit Map from Task 1 model coefficients
    deficit_weights = model_task1.coef_
    deficit_map_img = AtlasFeatureExtractor.memory_efficient_inverse_transform(
        deficit_weights, feature_extractor.atlas_img
    )
    deficit_map_img.to_filename(RESULTS_DIR / "deficit_map_atlas.nii.gz")
    print("Deficit map saved.")
    
    # Treatment Map from Task 2 model feature importances
    treatment_weights = model_task2.feature_importances_
    treatment_map_img = AtlasFeatureExtractor.memory_efficient_inverse_transform(
        treatment_weights, feature_extractor.atlas_img
    )
    
    # Enforce subset constraint (this logic remains the same)
    deficit_mask_img = nli.math_img("np.abs(img) > 1e-6", img=deficit_map_img)
    final_treatment_map_img = nli.math_img(
        "img1 * img2", img1=treatment_map_img, img2=deficit_mask_img
    )
    final_treatment_map_img.to_filename(RESULTS_DIR / "treatment_map_atlas.nii.gz")
    print("Treatment map saved.")

    print("\nAtlas model training script finished.")
    ```

#### 4. Acceptance Criteria
*   [ ] The `AtlasFeatureExtractor` class in `atlas.py` has a new static method `memory_efficient_inverse_transform`.
*   [ ] The `fit` method of `AtlasFeatureExtractor` is updated to load and store the atlas `Nifti1Image` object.
*   [ ] The `scripts/train_atlas_models.py` script is updated to use this new static method for generating both inference maps.
*   [ ] The `train_atlas_models.py` script runs to completion without any Out-of-Memory errors.
*   [ ] The generated `.nii.gz` files in the `results/` directory are valid and viewable.

#### 5. Testing Requirements
*   **Unit Tests:**
    *   Create `tests/features/test_atlas.py`.
    *   Write a new test case for `memory_efficient_inverse_transform`.
        *   Create a small dummy 3D numpy array for an atlas (e.g., 4x4x4 with 3 regions).
        *   Create a dummy `Nifti1Image` from it.
        *   Create a dummy weight vector (e.g., `np.array([10, 20, 30])`).
        *   Call the function and get the output `Nifti1Image`.
        *   Get the data from the output image and assert that the voxels corresponding to region 1 have the value 10, region 2 has 20, etc. This verifies the core logic.
*   **Manual Verification:** The primary validation is to run the full `scripts/train_atlas_models.py` script and confirm that it no longer crashes due to memory errors.

#### 6. Definition of Done
*   [ ] All Acceptance Criteria are met.
*   [ ] All required unit tests are written and pass.
*   [ ] The `train_atlas_models.py` script has been successfully run end-to-end on the full dataset.
*   [ ] The code has been formatted (`black`), linted (`ruff`), and type-checked (`mypy`).
*   [ ] The code has been peer-reviewed and approved.