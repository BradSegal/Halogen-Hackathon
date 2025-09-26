### Ticket: INF-01: Implement CNN Explanability and Inference Map Generation

#### 1. Description
This ticket details the implementation of the feature attribution pipeline required to solve Task 3 (Inference) using our trained `MultiTaskCNN`. A predictive model has limited value in a scientific context without interpretation. The goal of this ticket is to "look inside the black box" and extract the learned anatomical knowledge from the model.

We will use the **Integrated Gradients** algorithm, a state-of-the-art feature attribution method, to generate voxel-level saliency maps. These maps will highlight which anatomical locations the model deems most important for its predictions.

The process will involve three main parts:
1.  Integrating the `captum` library, a PyTorch-native library for model interpretability.
2.  Creating a robust, reusable function to generate a 3D saliency map for a single patient prediction from either the "severity" or "outcome" head of the `MultiTaskCNN`.
3.  Developing a script to systematically generate these individual saliency maps for relevant patient cohorts from the test set and aggregate them into final, population-level inference maps for the deficit substrate and the treatment-response substrate.

#### 2. Justification
This ticket is critical for fulfilling the project's core inference requirement and is grounded in our engineering principles:

*   **Principle of Least Surprise (POLS):** We are using Integrated Gradients, an axiomatic, standard method for feature attribution in deep learning. Its properties are well-understood, and it reliably attributes model predictions back to input features, avoiding the noise and saturation issues of simpler methods like vanilla gradients. By using a widely adopted library (`captum`), we ensure we are using a correct and standard implementation, making our results defensible.
*   **DRY (Don't Repeat Yourself):** We will create a single, reusable `generate_saliency_map` function. This function will be generic enough to be applied to either head of the `MultiTaskCNN` ("severity" or "outcome"), avoiding code duplication. The aggregation script will consume the `test.csv` file from `CORE-01`, building upon our validated data foundation.
*   **Strict Contracts:** The `captum` library has a clear API that we will adhere to. Our wrapper function will have a strict contract: accept a trained model, an input tensor, a treatment tensor, and a target head name, and it must return a 3D NumPy array representing the saliency map. This ensures predictable and testable behavior.

#### 3. Implementation Plan

**Step 1: Add `captum` as a Project Dependency**

1.  Open the `pyproject.toml` file.
2.  Add `"captum"` to the `dependencies` list.
3.  From your terminal, run `poetry lock && poetry install` to update your virtual environment.

**Step 2: Create a Reusable Explanability Module**

1.  Create a new file: `src/lesion_analysis/models/explanability.py`.
2.  Implement the core logic for generating an Integrated Gradients map for a single patient. The use of a `model_forward_wrapper` is a critical and non-obvious step required by `captum` for models that accept multiple inputs.

    ```python
    # File: src/lesion_analysis/models/explanability.py
    
    import torch
    import numpy as np
    from captum.attr import IntegratedGradients
    from src.lesion_analysis.models.cnn import MultiTaskCNN
    
    def generate_saliency_map(
        model: MultiTaskCNN,
        input_tensor: torch.Tensor,
        treatment_tensor: torch.Tensor,
        target_head: str,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Generates a 3D saliency map for a single input using Integrated Gradients.
    
        Args:
            model: The trained MultiTaskCNN model, already on the correct device.
            input_tensor: The lesion image tensor, shape (1, 1, 91, 109, 91), on device.
            treatment_tensor: The treatment assignment tensor, shape (1,), on device.
            target_head: The prediction head to explain ('severity' or 'outcome').
            n_steps: Number of steps for the path integral approximation.
    
        Returns:
            A 3D numpy array with voxel-level attributions.
        """
        if target_head not in ["severity", "outcome"]:
            raise ValueError("target_head must be either 'severity' or 'outcome'")
            
        model.eval()
        input_tensor.requires_grad = True
        
        # This wrapper is essential. Captum's `attribute` method requires a function
        # that takes only the input tensor we want to attribute to (the image).
        # Our model's forward pass takes two arguments (x, w). This wrapper
        # creates a new function that "freezes" the treatment_tensor argument.
        def model_forward_wrapper(img_tensor: torch.Tensor):
            return model(img_tensor, treatment_tensor)
        
        ig = IntegratedGradients(model_forward_wrapper)
        
        # Define the target for attribution. This tells captum which output neuron
        # to trace the gradients from. We get the output dictionary from a forward
        # pass and then specify the key corresponding to our desired head.
        def get_target_output(model_output: dict):
            return model_output[target_head]

        # Baseline is a healthy brain (an all-zeros tensor).
        baseline = torch.zeros_like(input_tensor)
        
        # Generate attributions. The `attribute` method returns a tensor of the same
        # shape as the input.
        attributions, delta = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=get_target_output,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        
        # The convergence delta should be close to zero, indicating the approximation is accurate.
        print(f"IG Convergence Delta for {target_head}: {delta.item():.4f}")
        
        # Return the attributions as a 3D numpy array, moved to the CPU.
        return attributions.squeeze().cpu().detach().numpy()
    ```

**Step 3: Create the Inference Map Generation Script**

1.  Create a new script: `scripts/generate_cnn_maps.py`.
2.  This script will load the trained model, define the patient cohorts for analysis, iterate through them to generate saliency maps, and average the results.

    ```python
    # File: scripts/generate_cnn_maps.py
    
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
        deficit_threshold = train_df[train_df.clinical_score > 0].clinical_score.quantile(0.75)
        deficit_cohort_df = test_df[test_df.clinical_score >= deficit_threshold]
        if args.smoke_test:
            deficit_cohort_df = deficit_cohort_df.head(2)
        print(f"Identifying deficit map from {len(deficit_cohort_df)} high-deficit patients...")
        
        all_deficit_maps = []
        for _, row in tqdm(deficit_cohort_df.iterrows(), total=len(deficit_cohort_df), desc="Deficit Map Saliency"):
            img = nib.load(row['lesion_filepath'])
            img_tensor = torch.from_numpy(img.get_fdata(dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            w_tensor = torch.tensor([0], dtype=torch.float32).to(device) # W is irrelevant for severity head
            
            saliency_map = generate_saliency_map(model, img_tensor, w_tensor, target_head="severity")
            all_deficit_maps.append(saliency_map)
            
        mean_deficit_map = np.mean(all_deficit_maps, axis=0)
    
        # --- Generate Treatment Map (Task 3b) ---
        print("\n--- Generating Treatment Response Substrate Map ---")
        # Cohort: Patients in the test set who received treatment and are PREDICTED to be responders.
        treatment_cohort_df = test_df[test_df.treatment_assignment == "Treatment"].copy()
        
        print(f"Calculating predicted CATE for {len(treatment_cohort_df)} treated patients...")
        cates = []
        with torch.no_grad():
            for _, row in treatment_cohort_df.iterrows():
                img = nib.load(row['lesion_filepath'])
                img_tensor = torch.from_numpy(img.get_fdata(dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict outcome with treatment
                pred_treated = model(img_tensor, torch.tensor([1], dtype=torch.float32).to(device))
                # Predict outcome without treatment (counterfactual)
                pred_control = model(img_tensor, torch.tensor([0], dtype=torch.float32).to(device))
                
                # CATE = improvement = outcome_control - outcome_treated
                cate = pred_control["outcome"].item() - pred_treated["outcome"].item()
                cates.append(cate)
        
        treatment_cohort_df['predicted_cate'] = cates
        responder_cohort_df = treatment_cohort_df[treatment_cohort_df['predicted_cate'] > 0]
        if args.smoke_test:
            responder_cohort_df = responder_cohort_df.head(2)
        print(f"Identifying treatment map from {len(responder_cohort_df)} predicted responders...")
    
        all_treatment_maps = []
        for _, row in tqdm(responder_cohort_df.iterrows(), total=len(responder_cohort_df), desc="Treatment Map Saliency"):
            img = nib.load(row['lesion_filepath'])
            img_tensor = torch.from_numpy(img.get_fdata(dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            w_tensor = torch.tensor([1], dtype=torch.float32).to(device) # Must explain outcome for treated patient
            
            saliency_map = generate_saliency_map(model, img_tensor, w_tensor, target_head="outcome")
            all_treatment_maps.append(saliency_map)
    
        mean_treatment_map = np.mean(all_treatment_maps, axis=0) if all_treatment_maps else np.zeros_like(mean_deficit_map)
    
        # --- Save Maps ---
        print("\nSaving inference maps...")
        sample_affine = nib.load(test_df.iloc[0]['lesion_filepath']).affine
        
        nib.save(nib.Nifti1Image(mean_deficit_map, sample_affine), RESULTS_DIR / "deficit_map_cnn.nii.gz")
        nib.save(nib.Nifti1Image(mean_treatment_map, sample_affine), RESULTS_DIR / "treatment_map_cnn.nii.gz")
        print("Inference maps saved successfully.")
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--smoke-test", action="store_true", help="Run on a small subset for testing.")
        args = parser.parse_args()
        main(args)
    ```

#### 4. Acceptance Criteria
*   [ ] The `captum` library is successfully added as a project dependency.
*   [ ] The file `src/lesion_analysis/models/explanability.py` exists and contains the `generate_saliency_map` function as specified.
*   [ ] The script `scripts/generate_cnn_maps.py` runs to completion from the command line without errors.
*   [ ] The script generates and saves two NIfTI files in the `results/` directory: `deficit_map_cnn.nii.gz` and `treatment_map_cnn.nii.gz`.
*   [ ] The generated `.nii.gz` files are 3D volumes with the correct dimensions `(91, 109, 91)` and can be opened in a standard NIfTI viewer.

#### 5. Testing Requirements
*   **Unit Tests:**
    *   Create `tests/models/test_explanability.py`.
    *   **Test `generate_saliency_map`:**
        *   Instantiate a `MultiTaskCNN` model (it does not need to be trained for this test).
        *   Create a dummy input tensor, a dummy treatment tensor, and a dummy baseline tensor.
        *   Mock the `captum.attr.IntegratedGradients` object to avoid running the expensive computation. Configure the mock to return a tensor of the correct shape.
        *   Call the `generate_saliency_map` function and assert that the returned NumPy array has the correct shape `(91, 109, 91)`.
        *   Assert that calling the function with an invalid `target_head` raises a `ValueError`.
*   **Integration Tests (Smoke Test):**
    *   Create `tests/scripts/test_generate_cnn_maps.py`.
    *   The test will execute the script `scripts/generate_cnn_maps.py` as a subprocess with the `--smoke-test` flag.
    *   This test will use a pre-trained dummy model checkpoint (which can be created by the test fixture itself) to ensure the script doesn't depend on a full, successful training run.
    *   The test must assert that both output `.nii.gz` files are created in the correct location, even if they are based on a cohort of only 2 patients.

#### 6. Definition of Done
*   [ ] All Acceptance Criteria are met.
*   [ ] All required unit and smoke tests are written and pass.
*   [ ] The code has been formatted (`black`), linted (`ruff`), and type-checked (`mypy`) successfully.
*   [ ] The project `README.md` is updated to describe the CNN inference procedure and how to run the generation script.
*   [ ] The code has been peer-reviewed and approved.