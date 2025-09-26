#!/usr/bin/env python3
"""
Multi-task CNN Training Script for Brain Lesion Analysis.

This script trains a multi-task 3D CNN with a shared backbone for simultaneous
prediction of clinical severity and treatment outcomes, using conditional
backpropagation to handle partially available labels.

Usage:
    python scripts/train_multitask_cnn.py
    python scripts/train_multitask_cnn.py --smoke-test  # For testing
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lesion_analysis.models.cnn import MultiTaskCNN
from lesion_analysis.models.torch_loader import LesionDataset
from lesion_analysis.models.loss import conditional_multitask_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_data_loaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    smoke_test: bool = False,
):
    """
    Set up train and validation data loaders for multi-task learning.

    Parameters
    ----------
    train_path : str
        Path to the train.csv file
    val_path : str
        Path to the validation.csv file
    batch_size : int
        Batch size for data loaders
    smoke_test : bool
        If True, use only a small subset of data for testing

    Returns
    -------
    tuple
        (train_loader, val_loader, train_stats) where train_stats contains dataset statistics
    """
    # Load canonical train and validation sets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Log dataset statistics
    train_severity_samples = train_df["clinical_score"].notna().sum()
    train_outcome_samples = train_df["outcome_score"].notna().sum()
    val_severity_samples = val_df["clinical_score"].notna().sum()
    val_outcome_samples = val_df["outcome_score"].notna().sum()

    logger.info(f"Training set: {len(train_df)} total samples")
    logger.info(f"  - Severity labels: {train_severity_samples}")
    logger.info(f"  - Outcome labels: {train_outcome_samples}")
    logger.info(f"Validation set: {len(val_df)} total samples")
    logger.info(f"  - Severity labels: {val_severity_samples}")
    logger.info(f"  - Outcome labels: {val_outcome_samples}")

    # For smoke test, use only a small subset
    if smoke_test:
        train_df = train_df.head(16)  # Use only 16 samples for smoke test
        val_df = val_df.head(4)  # Use only 4 samples for validation in smoke test
        logger.info(
            f"Smoke test mode: Using only {len(train_df)} train and {len(val_df)} validation samples"
        )

    # Create datasets from the DataFrames
    train_dataset = LesionDataset(train_df)
    val_dataset = LesionDataset(val_df)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    train_stats = {
        "n_severity": train_severity_samples,
        "n_outcome": train_outcome_samples,
    }

    return train_loader, val_loader, train_stats


def train_epoch(
    model, train_loader, optimizer, device, outcome_weight=1.0, smoke_test=False
):
    """Train for one epoch with multi-task learning."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, targets in pbar:
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
        loss = conditional_multitask_loss(
            predictions, targets_on_device, outcome_weight
        )

        # --- BACKWARD PASS & OPTIMIZER STEP ---
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # For smoke test, only process one batch
        if smoke_test:
            break

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, device, outcome_weight=1.0, smoke_test=False):
    """Validate for one epoch with multi-task learning."""
    model.eval()
    total_loss = 0.0
    all_severity_preds = []
    all_severity_targets = []
    all_outcome_preds = []
    all_outcome_targets = []
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, targets in pbar:
            images = images.to(device)
            treatment_w = targets["treatment"].to(device)
            targets_on_device = {k: v.to(device) for k, v in targets.items()}

            # Forward pass
            predictions = model(images, treatment_w)

            # Calculate loss
            loss = conditional_multitask_loss(
                predictions, targets_on_device, outcome_weight
            )

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets for metrics
            all_severity_preds.extend(predictions["severity"].cpu().numpy())
            all_severity_targets.extend(targets["severity"].cpu().numpy())
            all_outcome_preds.extend(predictions["outcome"].cpu().numpy())
            all_outcome_targets.extend(targets["outcome"].cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # For smoke test, only process one batch
            if smoke_test:
                break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate RMSE for severity (all samples should have severity labels)
    severity_rmse = np.sqrt(
        mean_squared_error(all_severity_targets, all_severity_preds)
    )

    # Calculate RMSE for outcome (only on samples with valid outcome labels)
    outcome_targets_array = np.array(all_outcome_targets)
    outcome_preds_array = np.array(all_outcome_preds)
    valid_outcome_mask = ~np.isnan(outcome_targets_array)

    if valid_outcome_mask.any():
        outcome_rmse = np.sqrt(
            mean_squared_error(
                outcome_targets_array[valid_outcome_mask],
                outcome_preds_array[valid_outcome_mask],
            )
        )
    else:
        outcome_rmse = np.nan

    return avg_loss, severity_rmse, outcome_rmse


def train_model(smoke_test: bool = False):
    """
    Train a multi-task CNN model.

    Parameters
    ----------
    smoke_test : bool
        If True, run minimal training for testing purposes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data loading
    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"
    train_loader, val_loader, train_stats = setup_data_loaders(
        train_path, val_path, smoke_test=smoke_test
    )

    # Model setup
    model = MultiTaskCNN(dropout_rate=0.5).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # OPTIMAL CONFIGURATION: AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # OPTIMAL CONFIGURATION: Learning Rate Scheduler
    # For multi-task, we monitor the combined validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # Training parameters
    max_epochs = 2 if smoke_test else 50
    patience = 1 if smoke_test else 5
    best_combined_metric = float("inf")  # Lower is better for RMSE
    best_epoch = 0
    epochs_without_improvement = 0
    outcome_weight = 1.0  # Weight for outcome loss

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = "models/multitask_cnn_model.pt"

    logger.info(
        f"Starting multi-task training (max_epochs={max_epochs}, patience={patience})"
    )

    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, device, outcome_weight, smoke_test
        )

        # Validation
        val_loss, severity_rmse, outcome_rmse = validate_epoch(
            model, val_loader, device, outcome_weight, smoke_test
        )

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Severity RMSE: {severity_rmse:.4f}")
        if not np.isnan(outcome_rmse):
            logger.info(f"Val Outcome RMSE: {outcome_rmse:.4f}")

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

        # For early stopping, use a combined metric (average of RMSEs)
        if not np.isnan(outcome_rmse):
            combined_metric = (severity_rmse + outcome_rmse) / 2
        else:
            combined_metric = severity_rmse

        # Early stopping logic
        if combined_metric < best_combined_metric:
            best_combined_metric = combined_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            logger.info(
                f"New best model saved! Best combined metric: {best_combined_metric:.4f}"
            )
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping after {epoch + 1} epochs (best epoch: {best_epoch + 1})"
            )
            break

    logger.info(f"Training completed. Best combined metric: {best_combined_metric:.4f}")
    logger.info(f"Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-task CNN model for brain lesion analysis"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal training for testing purposes",
    )

    args = parser.parse_args()

    # Run training without broad exception handler to allow errors to propagate
    train_model(args.smoke_test)
    logger.info("Multi-task training completed successfully!")


if __name__ == "__main__":
    main()
