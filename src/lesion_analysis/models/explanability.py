import torch
import numpy as np
from captum.attr import IntegratedGradients
from src.lesion_analysis.models.cnn import MultiTaskCNN


def generate_saliency_map(
    model: MultiTaskCNN,
    input_tensor: torch.Tensor,
    treatment_tensor: torch.Tensor,
    target_head: str,
    n_steps: int = 50,
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
        # Expand treatment tensor to match batch size for integrated gradients
        batch_size = img_tensor.shape[0]
        treatment_expanded = treatment_tensor.expand(batch_size)
        output = model(img_tensor, treatment_expanded)
        return output[target_head]

    ig = IntegratedGradients(model_forward_wrapper)

    # Baseline is a healthy brain (an all-zeros tensor).
    baseline = torch.zeros_like(input_tensor)

    # Generate attributions. The `attribute` method returns a tensor of the same
    # shape as the input.
    attributions = ig.attribute(
        input_tensor, baselines=baseline, n_steps=n_steps, return_convergence_delta=True
    )

    # Handle the tuple return from attribute method
    if isinstance(attributions, tuple):
        attributions, delta = attributions
        print(f"IG Convergence Delta for {target_head}: {delta.item():.4f}")

    # Return the attributions as a 3D numpy array, moved to the CPU.
    return attributions.squeeze().cpu().detach().numpy()
