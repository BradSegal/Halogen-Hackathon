### Ticket: TASK-03.5: Evolve CNN to a Multi-Task Model with a Shared Backbone and Conditional Inputs

#### 1. Description
This ticket details a critical evolution of our CNN approach, upgrading it from a single-task model to a more powerful and data-efficient **multi-task learning** architecture. The new model will feature a single, shared convolutional "backbone" that learns a rich spatial representation from all available lesion maps. This backbone will feed its feature vector into two separate prediction "heads":

1.  **Severity Head:** A purely prognostic head that predicts the initial `clinical_score` based only on the lesion anatomy.
2.  **Outcome Head:** A conditional head that predicts the `outcome_score`. Critically, this head will receive both the lesion anatomy features *and* the treatment assignment variable (`W`) as input, allowing it to learn treatment-specific effects.

Training is accomplished via a custom loss function that implements "conditional backpropagation": for any given sample, gradients are only computed and propagated from the heads for which a ground-truth label exists. This approach allows the model's core feature extractor to learn from the entire dataset (all 4119 samples), dramatically improving its ability to generalize, especially for the data-sparse outcome prediction task. This architecture also allows for the direct estimation of the Conditional Average Treatment Effect (CATE) during inference.

#### 2. Justification
This multi-task approach is a direct and sophisticated application of our core principles, chosen to maximize the performance and utility of our deep learning model:

*   **DRY (Don't Repeat Yourself):** This is the epitome of DRY at the model level. Instead of training two separate, data-hungry CNNs, we train a single, shared feature extractor. The knowledge gained from learning to predict `clinical_score` from all ~4k samples is directly transferred and reused for the more difficult task of predicting `outcome_score` from a smaller subset. This prevents redundant learning and makes the absolute most of our limited data.
*   **Principle of Least Surprise (POLS):** The multi-headed architecture is a standard, well-understood pattern for multi-task learning. The implementation of "conditional backprop" via a masked loss function is the canonical way to handle tasks with partially available labels in PyTorch. The explicit concatenation of the treatment variable (`W`) into the outcome head is the most direct and causally correct way to model conditional outcomes, making the architecture's logic transparent and defensible.
*   **Strict Contracts:** The new model will have a clear, updated contract. Its `forward` method will accept an image tensor `x` and a treatment tensor `w`, and it will return a dictionary of predictions (`{'severity': ..., 'outcome': ...}`). The custom loss function has a strict contract to accept this prediction dictionary and a corresponding target dictionary, correctly handling missing labels.

#### 3. Implementation Plan

**Step 1: Update the PyTorch `Dataset` to Provide All Necessary Data**

1.  Modify the file `src/lesion_analysis/models/torch_loader.py`.
2.  The `__getitem__` method must be updated to return a dictionary of targets, including the treatment variable `W` (encoded as 0 for Control, 1 for Treatment) and using `np.nan` for missing labels.

    ```python
    # File: src/lesion_analysis/models/torch_loader.py (Updated section)
    
    import torch
    import pandas as pd
    import numpy as np
    import nibabel as nib
    from torch.utils.data import Dataset
    from typing import Tuple, Dict
    
    class LesionDataset(Dataset):
        # ... __init__ and __len__ are unchanged ...
    
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            record = self.df.iloc[idx]
            filepath = record["lesion_filepath"]
            
            img = nib.load(filepath)
            data = torch.from_numpy(img.get_fdata(dtype=np.float32)).unsqueeze(0) # Shape: (1, 91, 109, 91)
            
            # Encode treatment assignment 'W'. Control=0, Treatment=1. 
            # If 'treatment_assignment' is NaN/None, default to 0 (it won't be used for outcome loss anyway).
            treatment_val = 1 if record["treatment_assignment"] == "Treatment" else 0
            
            targets = {
                "severity": torch.tensor(record["clinical_score"], dtype=torch.float32),
                "outcome": torch.tensor(record.get("outcome_score", np.nan), dtype=torch.float32),
                "treatment": torch.tensor(treatment_val, dtype=torch.float32)
            }
            
            return data, targets
    ```

**Step 2: Implement the Multi-Headed CNN Architecture**

1.  Modify the file `src/lesion_analysis/models/cnn.py`.
2.  Refactor the `Simple3DCNN` into the new, causally-correct `MultiTaskCNN` class. Note the change in the `outcome_head`'s input dimension and the `forward` method's signature.

    ```python
    # File: src/lesion_analysis/models/cnn.py (Updated/New)
    
    import torch
    import torch.nn as nn
    from typing import Dict
    
    class MultiTaskCNN(nn.Module):
        def __init__(self, in_channels: int = 1, backbone_features: int = 32, dropout_rate: float = 0.5):
            super().__init__()
            # Shared Backbone extracts a feature vector from the 3D lesion map
            self.backbone = nn.Sequential(
                nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
                nn.BatchNorm3d(8), nn.ReLU(), nn.MaxPool3d(2), nn.Dropout3d(dropout_rate),
                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2), nn.Dropout3d(dropout_rate),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2), nn.Dropout3d(dropout_rate),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten()
            )
            
            # Head for predicting initial severity (prognostic)
            self.severity_head = nn.Linear(backbone_features, 1)
            
            # Head for predicting outcome, conditioned on anatomy AND treatment
            self.outcome_head = nn.Linear(backbone_features + 1, 1) # +1 for the treatment variable
    
        def forward(self, x: torch.Tensor, w: torch.Tensor) -> Dict[str, torch.Tensor]:
            # x: image tensor, shape (batch, 1, 91, 109, 91)
            # w: treatment tensor, shape (batch,)
            features = self.backbone(x)
            
            # Severity prediction is unconditional on treatment
            pred_severity = self.severity_head(features).squeeze(-1)
            
            # For outcome, concatenate anatomical features with treatment variable
            outcome_head_input = torch.cat([features, w.unsqueeze(1)], dim=1)
            pred_outcome = self.outcome_head(outcome_head_input).squeeze(-1)
            
            return {"severity": pred_severity, "outcome": pred_outcome}
    ```

**Step 3: Implement the Custom Conditional Loss Function**

1.  Create a new file for organized loss functions: `src/lesion_analysis/models/loss.py`.
2.  Add the conditional loss function which handles missing labels gracefully.

    ```python
    # File: src/lesion_analysis/models/loss.py
    
    import torch
    import torch.nn.functional as F
    
    def conditional_multitask_loss(
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor], 
        outcome_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Calculates a combined MSE loss for severity and outcome, but only computes
        the outcome loss for samples that have a valid (non-NaN) outcome label.
        """
        # --- Severity Loss (computed for all samples in the batch) ---
        loss_severity = F.mse_loss(predictions["severity"], targets["severity"])
        
        # --- Outcome Loss (computed conditionally) ---
        outcome_preds = predictions["outcome"]
        outcome_targets = targets["outcome"]
        
        # Create a boolean mask for valid (non-NaN) outcome targets
        valid_mask = ~torch.isnan(outcome_targets)
        
        # Only compute loss if there are any valid targets in the batch
        if valid_mask.any():
            loss_outcome = F.mse_loss(
                outcome_preds[valid_mask], 
                outcome_targets[valid_mask]
            )
        else:
            # If no valid targets in this batch, outcome loss is zero
            loss_outcome = torch.tensor(0.0, device=loss_severity.device, dtype=loss_severity.dtype)
            
        total_loss = loss_severity + outcome_weight * loss_outcome
        return total_loss
    ```

**Step 4: Update the Training Script to a Multi-Task Loop**

1.  Rename `scripts/train_cnn_models.py` to `scripts/train_multitask_cnn.py`.
2.  Refactor the training loop to use the new multi-task components.

    ```python
    # In scripts/train_multitask_cnn.py, key modifications inside the training loop:
    
    # ... inside the loop over epochs and batches ...
    for images, targets in train_loader:
        images = images.to(device)
        # Unpack the treatment variable and move all targets to the correct device
        treatment_w = targets["treatment"].to(device)
        targets_on_device = {k: v.to(device) for k, v in targets.items()}
    
        optimizer.zero_grad()
        
        # --- FORWARD PASS ---
        # Pass both image and treatment assignment to the model
        predictions = model(images, treatment_w)
        
        # --- LOSS CALCULATION ---
        # The custom loss function handles the logic internally
        loss = conditional_multitask_loss(predictions, targets_on_device, outcome_weight=1.0)
        
        # --- BACKWARD PASS & OPTIMIZER STEP ---
        loss.backward()
        optimizer.step()
    # ...
    ```

#### 4. Acceptance Criteria
*   [ ] The `LesionDataset` in `torch_loader.py` is updated to return a dictionary of targets that includes the `treatment` variable.
*   [ ] The file `cnn.py` contains the `MultiTaskCNN` class with a shared backbone, a `severity_head` taking `N` features, and an `outcome_head` taking `N+1` features.
*   [ ] The `forward` method of `MultiTaskCNN` accepts both `x` (image) and `w` (treatment) tensors.
*   [ ] The file `loss.py` exists and contains the `conditional_multitask_loss` function.
*   [ ] The script `scripts/train_multitask_cnn.py` runs to completion and saves a single best-performing model checkpoint: `models/multitask_cnn_model.pt`.
*   [ ] The training log shows a single combined loss value. The validation step should report separate RMSE metrics for severity and outcome (calculated on the relevant subset).

#### 5. Testing Requirements
*   **Unit Tests:**
    *   **Test `MultiTaskCNN` (`tests/models/test_cnn.py`):**
        *   Instantiate `MultiTaskCNN`. Pass a dummy image tensor `x` and a dummy treatment tensor `w`. Assert that the output is a dictionary with keys `"severity"` and `"outcome"`, and that the value tensors have the correct shape `(batch_size,)`.
    *   **Test `LesionDataset` (`tests/models/test_torch_loader.py`):**
        *   Assert that `dataset[idx]` returns a tuple where the second element is a dictionary containing the keys `"severity"`, `"outcome"`, and `"treatment"`.
    *   **Test `conditional_multitask_loss` (`tests/models/test_loss.py`):**
        *   **Case 1 (Full Labels):** Pass predictions and targets where all labels are present. Assert the loss is `mse(pred_sev, tar_sev) + w*mse(pred_out, tar_out)`.
        *   **Case 2 (Partial Labels):** Pass a batch where some `outcome` targets are `NaN`. Assert the function runs and the loss is computed correctly on the valid subset.
        *   **Case 3 (No Outcome Labels):** Pass a batch where all `outcome` targets are `NaN`. Assert that the total loss is exactly equal to the severity loss.
*   **Integration Tests (Smoke Test):**
    *   Update the smoke test in `tests/scripts/` to run `scripts/train_multitask_cnn.py`.
    *   The test must verify that the new multi-task training loop—including unpacking `w` from the data loader, the `model(x, w)` forward pass, and the custom loss calculation—executes for one batch without crashing.

#### 6. Definition of Done
*   [ ] All Acceptance Criteria are met.
*   [ ] All required unit and smoke tests are written and pass.
*   [ ] The code has been formatted (`black`), linted (`ruff`), and type-checked (`mypy`) successfully.
*   [ ] The project `README.md` and/or `CLAUDE.md` is updated to describe the multi-task CNN and its training procedure.
*   [ ] The code has been peer-reviewed and approved.