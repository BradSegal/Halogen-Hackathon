#!/usr/bin/env python3
"""
CNN Training Script for Brain Lesion Analysis.

This script trains 3D CNNs for both Task 1 (regression) and Task 2 (classification)
with aggressive regularization to prevent overfitting on the small dataset.

Usage:
    python scripts/train_cnn_models.py --task task1
    python scripts/train_cnn_models.py --task task2
    python scripts/train_cnn_models.py --task task1 --smoke-test  # For testing
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from tqdm import tqdm
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lesion_analysis.models.cnn import Simple3DCNN
from lesion_analysis.models.torch_loader import LesionDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_data_loaders(
    train_path: str,
    val_path: str,
    task: str,
    batch_size: int = 32,
    smoke_test: bool = False,
    train_on_all: bool = False,
):
    """
    Set up train and validation data loaders for the specified task.

    Parameters
    ----------
    train_path : str
        Path to the train.csv file
    val_path : str
        Path to the validation.csv file
    task : str
        Either 'task1' or 'task2'
    batch_size : int
        Batch size for data loaders
    smoke_test : bool
        If True, use only a small subset of data for testing
    train_on_all : bool
        If True, train Task 1 on all data including zero scores

    Returns
    -------
    tuple
        (train_loader, val_loader, pos_weight) where pos_weight is None for task1
    """
    # Load canonical train and validation sets
    train_df_full = pd.read_csv(train_path)
    val_df_full = pd.read_csv(val_path)

    if task == "task1":
        # Task 1: Regression - conditionally filter based on train_on_all flag
        if train_on_all:
            train_df = train_df_full.copy()
            val_df = val_df_full.copy()
            logger.info(
                "Task 1: Using ALL training and validation samples for regression"
            )
        else:
            train_df = train_df_full[train_df_full["clinical_score"] > 0].copy()
            val_df = val_df_full[val_df_full["clinical_score"] > 0].copy()
            logger.info("Task 1: Using non-zero score samples for regression")
        target_col = "clinical_score"
        logger.info(
            f"Task 1: Using {len(train_df)} train, {len(val_df)} validation samples"
        )
        pos_weight = None
    elif task == "task2":
        # Task 2: Classification - filter for samples with treatment assignment
        train_df = train_df_full[train_df_full["treatment_assignment"].notna()].copy()
        val_df = val_df_full[val_df_full["treatment_assignment"].notna()].copy()
        target_col = "is_responder"
        logger.info(
            f"Task 2: Using {len(train_df)} train, {len(val_df)} validation samples for classification"
        )

        # Calculate pos_weight for class imbalance using training set only
        num_positive = train_df["is_responder"].sum()
        num_negative = len(train_df) - num_positive
        pos_weight = (
            torch.tensor(num_negative / num_positive)
            if num_positive > 0
            else torch.tensor(1.0)
        )
        logger.info(
            f"Train class balance: {num_positive} positive, {num_negative} negative, pos_weight: {pos_weight:.3f}"
        )
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'task1' or 'task2'")

    # For smoke test, use only a small subset
    # if smoke_test:
    train_df = train_df.head(16)  # Use only 16 samples for smoke test
    val_df = val_df.head(4)  # Use only 4 samples for validation in smoke test
    logger.info(
        f"Smoke test mode: Using only {len(train_df)} train and {len(val_df)} validation samples"
    )

    # Create datasets from the DataFrames
    train_dataset = LesionDataset(train_df, target_col)
    val_dataset = LesionDataset(val_df, target_col)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(
        f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    return train_loader, val_loader, pos_weight


def train_epoch(model, train_loader, criterion, optimizer, device, smoke_test=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_data, batch_targets in pbar:
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # For smoke test, only process one batch
        if smoke_test:
            break

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, criterion, device, task, smoke_test=False):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_data, batch_targets in pbar:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)

            total_loss += loss.item()
            num_batches += 1

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # For smoke test, only process one batch
            if smoke_test:
                break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Calculate task-specific metrics
    if task == "task1":
        # Task 1: RMSE for regression
        rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
        metric_value = rmse
        metric_name = "RMSE"
    else:
        # Task 2: Balanced Accuracy for classification
        predictions = (torch.sigmoid(torch.tensor(all_outputs)) > 0.5).numpy()
        balanced_acc = balanced_accuracy_score(all_targets, predictions)
        metric_value = balanced_acc
        metric_name = "Balanced Accuracy"

    return avg_loss, metric_value, metric_name


def train_model(task: str, smoke_test: bool = False, train_on_all: bool = False):
    """
    Train a CNN model for the specified task.

    Parameters
    ----------
    task : str
        Either 'task1' or 'task2'
    smoke_test : bool
        If True, run minimal training for testing purposes
    train_on_all : bool
        If True, train Task 1 on all data including zero scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data loading
    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"
    train_loader, val_loader, pos_weight = setup_data_loaders(
        train_path, val_path, task, smoke_test=smoke_test, train_on_all=train_on_all
    )

    # Model setup
    model = Simple3DCNN(num_classes=1, dropout_rate=0).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion: Union[nn.MSELoss, nn.BCEWithLogitsLoss]
    if task == "task1":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device) if pos_weight else None
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # , weight_decay=1e-2)

    # Training parameters
    max_epochs = 2 if smoke_test else 50
    patience = 1 if smoke_test else 5
    # Initialize best_metric correctly for each task
    if task == "task1":
        best_metric = float("inf")  # Lower is better for RMSE
    else:
        best_metric = -1.0  # Higher is better for accuracy, init below valid range
    best_epoch = 0
    epochs_without_improvement = 0

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{task}_cnn_model.pt"

    logger.info(
        f"Starting training for {task} (max_epochs={max_epochs}, patience={patience})"
    )

    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{max_epochs}")

        # Training
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, smoke_test
        )

        # Validation
        val_loss, metric_value, metric_name = validate_epoch(
            model, val_loader, criterion, device, task, smoke_test
        )

        logger.info(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {metric_name}: {metric_value:.4f}"
        )

        # Early stopping logic
        improved = False
        if task == "task1":
            # For regression, lower RMSE is better
            if metric_value < best_metric:
                improved = True
                best_metric = metric_value
        else:
            # For classification, higher balanced accuracy is better
            if metric_value > best_metric:
                improved = True
                best_metric = metric_value

        if improved:
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved! Best {metric_name}: {best_metric:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping after {epoch + 1} epochs (best epoch: {best_epoch + 1})"
            )
            break

    logger.info(f"Training completed. Best {metric_name}: {best_metric:.4f}")
    logger.info(f"Model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN models for brain lesion analysis"
    )
    parser.add_argument(
        "--task",
        choices=["task1", "task2"],
        required=True,
        help="Task to train: task1 (regression) or task2 (classification)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal training for testing purposes",
    )
    parser.add_argument(
        "--train-on-all-data",
        action="store_true",
        help="Train Task 1 on all data, not just score > 0.",
    )

    args = parser.parse_args()

    # Run training without broad exception handler to allow errors to propagate
    train_model(args.task, args.smoke_test, args.train_on_all_data)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
