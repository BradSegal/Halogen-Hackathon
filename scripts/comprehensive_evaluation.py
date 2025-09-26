#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for Brain Lesion Analysis Models

This script provides a unified evaluation framework for all model types,
generating detailed metrics, per-item predictions, and visualizations.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.lesion_analysis.features.atlas import AtlasFeatureExtractor
from src.lesion_analysis.features.voxel import downsample_and_flatten_lesions
from src.lesion_analysis.models.cnn import Simple3DCNN, MultiTaskCNN
from src.lesion_analysis.models.torch_loader import LesionDataset
from src.visualization import BrainLesionVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_CACHE_DIR = PROCESSED_DATA_DIR / "features_cache"


class ComprehensiveEvaluator:
    """Orchestrates comprehensive model evaluation with metrics and visualizations."""

    def __init__(
        self,
        outcomes_path: Path,
        output_dir: Path,
        models_to_eval: List[str],
        visualize: bool = True,
        per_item: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            outcomes_path: Path to CSV file with ground truth outcomes
            output_dir: Directory for all evaluation outputs
            models_to_eval: List of model types to evaluate
            visualize: Whether to generate visualizations
            per_item: Whether to generate per-item predictions
        """
        self.outcomes_path = outcomes_path
        self.output_dir = output_dir
        self.models_to_eval = models_to_eval
        self.visualize = visualize
        self.per_item = per_item

        # Create output directory structure
        self.create_output_structure()

        # Load test data
        self.test_df = pd.read_csv(outcomes_path)
        logger.info(f"Loaded {len(self.test_df)} test samples from {outcomes_path}")

        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Results storage
        self.all_metrics = {}
        self.predictions = {}

    def create_output_structure(self):
        """Create the output directory structure."""
        subdirs = [
            "metrics",
            "predictions",
            "visualizations",
            "visualizations/prediction_scatter",
            "visualizations/error_analysis",
            "visualizations/brain_maps",
            "visualizations/confusion_matrices",
            "reports",
            "logs",
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Setup logging to file
        file_handler = logging.FileHandler(
            self.output_dir / "logs" / "evaluation.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    def calculate_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        metrics = {}

        # Basic metrics
        metrics[f"{prefix}RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f"{prefix}MAE"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{prefix}R2"] = r2_score(y_true, y_pred)

        # Correlation metrics
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        metrics[f"{prefix}Pearson_r"] = pearson_corr
        metrics[f"{prefix}Pearson_p"] = pearson_p
        metrics[f"{prefix}Spearman_r"] = spearman_corr
        metrics[f"{prefix}Spearman_p"] = spearman_p

        # Error statistics
        errors = y_true - y_pred
        metrics[f"{prefix}Mean_Error"] = np.mean(errors)
        metrics[f"{prefix}Std_Error"] = np.std(errors)
        metrics[f"{prefix}Median_Error"] = np.median(errors)
        metrics[f"{prefix}Max_Error"] = np.max(np.abs(errors))

        # Percentage errors
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100
            metrics[f"{prefix}MAPE"] = mape

        return metrics

    def calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, prefix: str = ""
    ) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        metrics = {}

        # Basic metrics
        metrics[f"{prefix}Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics[f"{prefix}Accuracy"] = accuracy_score(y_true, y_pred)
        metrics[f"{prefix}Precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}Recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f"{prefix}F1_Score"] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics[f"{prefix}Confusion_Matrix"] = cm.tolist()
        metrics[f"{prefix}True_Negatives"] = cm[0, 0] if cm.shape[0] > 1 else 0
        metrics[f"{prefix}False_Positives"] = cm[0, 1] if cm.shape[0] > 1 else 0
        metrics[f"{prefix}False_Negatives"] = cm[1, 0] if cm.shape[0] > 1 else 0
        metrics[f"{prefix}True_Positives"] = cm[1, 1] if cm.shape[0] > 1 else 0

        # ROC-AUC if probabilities are available
        if y_prob is not None:
            try:
                metrics[f"{prefix}ROC_AUC"] = roc_auc_score(y_true, y_prob)
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                metrics[f"{prefix}ROC_Curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")

        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics[f"{prefix}Classification_Report"] = report

        return metrics

    def evaluate_baseline_models(self) -> Tuple[Dict, Dict]:
        """Evaluate baseline models."""
        logger.info("Evaluating Baseline Models...")
        metrics = {}
        predictions = {"task1": {}, "task2": {}}

        try:
            # Task 1: Regression
            model_path = MODELS_DIR / "task1_baseline_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                df_t1 = self.test_df[self.test_df.clinical_score > 0].copy()
                X = downsample_and_flatten_lesions(df_t1, FEATURES_CACHE_DIR)
                y_pred = model.predict(X)
                y_true = df_t1.clinical_score.values

                # Calculate metrics
                task1_metrics = self.calculate_regression_metrics(
                    y_true, y_pred, prefix="Task1_"
                )
                metrics.update(task1_metrics)

                # Store predictions
                if self.per_item:
                    predictions["task1"]["lesion_ids"] = df_t1.lesion_id.tolist()
                    predictions["task1"]["true_values"] = y_true.tolist()
                    predictions["task1"]["predicted_values"] = y_pred.tolist()
                    predictions["task1"]["errors"] = (y_true - y_pred).tolist()

                logger.info(f"  Task 1 RMSE: {task1_metrics['Task1_RMSE']:.4f}")
            else:
                logger.warning(f"Baseline Task 1 model not found at {model_path}")

        except Exception as e:
            logger.error(f"Error evaluating Baseline Task 1: {e}")

        try:
            # Task 2: Classification
            model_path = MODELS_DIR / "task2_baseline_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                df_t2 = self.test_df[self.test_df.treatment_assignment.notna()].copy()
                X = downsample_and_flatten_lesions(df_t2, FEATURES_CACHE_DIR)
                y_pred = model.predict(X)
                y_true = df_t2.is_responder.astype(bool).values

                # Get probabilities if available
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]

                # Calculate metrics
                task2_metrics = self.calculate_classification_metrics(
                    y_true, y_pred, y_prob, prefix="Task2_"
                )
                metrics.update(task2_metrics)

                # Store predictions
                if self.per_item:
                    predictions["task2"]["lesion_ids"] = df_t2.lesion_id.tolist()
                    predictions["task2"]["true_values"] = y_true.tolist()
                    predictions["task2"]["predicted_values"] = y_pred.tolist()
                    if y_prob is not None:
                        predictions["task2"]["probabilities"] = y_prob.tolist()

                logger.info(f"  Task 2 Balanced Accuracy: {task2_metrics['Task2_Balanced_Accuracy']:.4f}")
            else:
                logger.warning(f"Baseline Task 2 model not found at {model_path}")

        except Exception as e:
            logger.error(f"Error evaluating Baseline Task 2: {e}")

        return metrics, predictions

    def evaluate_atlas_models(self) -> Tuple[Dict, Dict]:
        """Evaluate atlas-based models."""
        logger.info("Evaluating Atlas Models...")
        metrics = {}
        predictions = {"task1": {}, "task2": {}}

        try:
            feature_extractor = AtlasFeatureExtractor(
                n_rois=400, model_dir=MODELS_DIR
            )

            # Task 1: Regression
            model_path = MODELS_DIR / "task1_atlas_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                df_t1 = self.test_df[self.test_df.clinical_score > 0].copy()
                X = feature_extractor.transform(df_t1)
                y_pred = model.predict(X)
                y_true = df_t1.clinical_score.values

                # Calculate metrics
                task1_metrics = self.calculate_regression_metrics(
                    y_true, y_pred, prefix="Task1_"
                )
                metrics.update(task1_metrics)

                # Store predictions
                if self.per_item:
                    predictions["task1"]["lesion_ids"] = df_t1.lesion_id.tolist()
                    predictions["task1"]["true_values"] = y_true.tolist()
                    predictions["task1"]["predicted_values"] = y_pred.tolist()
                    predictions["task1"]["errors"] = (y_true - y_pred).tolist()

                logger.info(f"  Task 1 RMSE: {task1_metrics['Task1_RMSE']:.4f}")
            else:
                logger.warning(f"Atlas Task 1 model not found at {model_path}")

        except Exception as e:
            logger.error(f"Error evaluating Atlas Task 1: {e}")

        try:
            feature_extractor = AtlasFeatureExtractor(
                n_rois=400, model_dir=MODELS_DIR
            )

            # Task 2: Classification
            model_path = MODELS_DIR / "task2_atlas_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                df_t2 = self.test_df[self.test_df.treatment_assignment.notna()].copy()
                X = feature_extractor.transform(df_t2)
                y_pred = model.predict(X)
                y_true = df_t2.is_responder.astype(bool).values

                # Get probabilities if available
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]

                # Calculate metrics
                task2_metrics = self.calculate_classification_metrics(
                    y_true, y_pred, y_prob, prefix="Task2_"
                )
                metrics.update(task2_metrics)

                # Store predictions
                if self.per_item:
                    predictions["task2"]["lesion_ids"] = df_t2.lesion_id.tolist()
                    predictions["task2"]["true_values"] = y_true.tolist()
                    predictions["task2"]["predicted_values"] = y_pred.tolist()
                    if y_prob is not None:
                        predictions["task2"]["probabilities"] = y_prob.tolist()

                logger.info(f"  Task 2 Balanced Accuracy: {task2_metrics['Task2_Balanced_Accuracy']:.4f}")
            else:
                logger.warning(f"Atlas Task 2 model not found at {model_path}")

        except Exception as e:
            logger.error(f"Error evaluating Atlas Task 2: {e}")

        return metrics, predictions

    def evaluate_cnn_models(self) -> Tuple[Dict, Dict]:
        """Evaluate CNN models."""
        logger.info("Evaluating CNN Models...")
        metrics = {}
        predictions = {"task1": {}, "task2": {}}

        try:
            # Task 1: Regression
            model_path = MODELS_DIR / "task1_cnn_model.pt"
            if model_path.exists():
                model = Simple3DCNN().to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()

                df_t1 = self.test_df[self.test_df.clinical_score > 0].copy()
                dataset = LesionDataset(df_t1, "clinical_score")
                loader = DataLoader(dataset, batch_size=8, shuffle=False)

                all_preds, all_targets, all_ids = [], [], []
                with torch.no_grad():
                    for imgs, targets in tqdm(loader, desc="CNN Task 1"):
                        imgs = imgs.to(self.device)
                        preds = model(imgs)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.numpy())

                y_true = np.array(all_targets)
                y_pred = np.array(all_preds).flatten()

                # Calculate metrics
                task1_metrics = self.calculate_regression_metrics(
                    y_true, y_pred, prefix="Task1_"
                )
                metrics.update(task1_metrics)

                # Store predictions
                if self.per_item:
                    predictions["task1"]["lesion_ids"] = df_t1.lesion_id.tolist()
                    predictions["task1"]["true_values"] = y_true.tolist()
                    predictions["task1"]["predicted_values"] = y_pred.tolist()
                    predictions["task1"]["errors"] = (y_true - y_pred).tolist()

                logger.info(f"  Task 1 RMSE: {task1_metrics['Task1_RMSE']:.4f}")
            else:
                logger.warning(f"CNN Task 1 model not found at {model_path}")

        except Exception as e:
            logger.error(f"Error evaluating CNN Task 1: {e}")

        try:
            # Task 2: Classification
            model_path = MODELS_DIR / "task2_cnn_model.pt"
            if model_path.exists():
                model = Simple3DCNN().to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()

                df_t2 = self.test_df[self.test_df.treatment_assignment.notna()].copy()
                dataset = LesionDataset(df_t2, "is_responder")
                loader = DataLoader(dataset, batch_size=8, shuffle=False)

                all_preds, all_probs, all_targets = [], [], []
                with torch.no_grad():
                    for imgs, targets in tqdm(loader, desc="CNN Task 2"):
                        imgs = imgs.to(self.device)
                        logits = model(imgs)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds = (probs > 0.5).astype(int)
                        all_probs.extend(probs)
                        all_preds.extend(preds)
                        all_targets.extend(targets.numpy())

                y_true = np.array(all_targets).astype(bool)
                y_pred = np.array(all_preds).flatten()
                y_prob = np.array(all_probs).flatten()

                # Calculate metrics
                task2_metrics = self.calculate_classification_metrics(
                    y_true, y_pred, y_prob, prefix="Task2_"
                )
                metrics.update(task2_metrics)

                # Store predictions
                if self.per_item:
                    predictions["task2"]["lesion_ids"] = df_t2.lesion_id.tolist()
                    predictions["task2"]["true_values"] = y_true.tolist()
                    predictions["task2"]["predicted_values"] = y_pred.tolist()
                    predictions["task2"]["probabilities"] = y_prob.tolist()

                logger.info(f"  Task 2 Balanced Accuracy: {task2_metrics['Task2_Balanced_Accuracy']:.4f}")
            else:
                logger.warning(f"CNN Task 2 model not found at {model_path}")

        except Exception as e:
            logger.error(f"Error evaluating CNN Task 2: {e}")

        return metrics, predictions

    def evaluate_multitask_cnn(self) -> Tuple[Dict, Dict]:
        """Evaluate MultiTask CNN model."""
        logger.info("Evaluating MultiTask CNN Model...")
        metrics = {}
        predictions = {"task1": {}, "task2": {}}

        try:
            model_path = MODELS_DIR / "multitask_cnn_model.pt"
            if not model_path.exists():
                logger.warning(f"MultiTask CNN model not found at {model_path}")
                return metrics, predictions

            model = MultiTaskCNN().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()

            # Task 1: Severity prediction
            df_t1 = self.test_df[self.test_df.clinical_score > 0].copy()
            dataset_t1 = LesionDataset(df_t1, "clinical_score")
            loader_t1 = DataLoader(dataset_t1, batch_size=8, shuffle=False)

            all_preds_t1, all_targets_t1 = [], []
            with torch.no_grad():
                for imgs, targets in tqdm(loader_t1, desc="MultiTask Task 1"):
                    imgs = imgs.to(self.device)
                    # Use W=0 for severity prediction (shape: batch_size,)
                    w = torch.zeros(imgs.size(0)).to(self.device)
                    outputs = model(imgs, w)
                    all_preds_t1.extend(outputs["severity"].cpu().numpy())
                    all_targets_t1.extend(targets.numpy())

            y_true_t1 = np.array(all_targets_t1)
            y_pred_t1 = np.array(all_preds_t1).flatten()

            # Calculate Task 1 metrics
            task1_metrics = self.calculate_regression_metrics(
                y_true_t1, y_pred_t1, prefix="Task1_"
            )
            metrics.update(task1_metrics)

            # Store Task 1 predictions
            if self.per_item:
                predictions["task1"]["lesion_ids"] = df_t1.lesion_id.tolist()
                predictions["task1"]["true_values"] = y_true_t1.tolist()
                predictions["task1"]["predicted_values"] = y_pred_t1.tolist()
                predictions["task1"]["errors"] = (y_true_t1 - y_pred_t1).tolist()

            logger.info(f"  Task 1 RMSE: {task1_metrics['Task1_RMSE']:.4f}")

            # Task 2: Treatment response (using CATE)
            df_t2 = self.test_df[self.test_df.treatment_assignment.notna()].copy()
            dataset_t2 = LesionDataset(df_t2, "is_responder")
            loader_t2 = DataLoader(dataset_t2, batch_size=8, shuffle=False)

            all_cates, all_targets_t2 = [], []
            with torch.no_grad():
                for imgs, targets in tqdm(loader_t2, desc="MultiTask Task 2"):
                    imgs = imgs.to(self.device)

                    # Calculate CATE (shape: batch_size,)
                    w_treat = torch.ones(imgs.size(0)).to(self.device)
                    w_control = torch.zeros(imgs.size(0)).to(self.device)

                    outputs_treat = model(imgs, w_treat)
                    outputs_control = model(imgs, w_control)

                    # CATE = outcome_control - outcome_treated (improvement)
                    cate = outputs_control["outcome"] - outputs_treat["outcome"]
                    all_cates.extend(cate.cpu().numpy())
                    all_targets_t2.extend(targets.numpy())

            y_true_t2 = np.array(all_targets_t2).astype(bool)
            # Predict responder if CATE > 0
            y_pred_t2 = (np.array(all_cates).flatten() > 0).astype(int)
            y_prob_t2 = 1 / (1 + np.exp(-np.array(all_cates).flatten()))  # Sigmoid of CATE

            # Calculate Task 2 metrics
            task2_metrics = self.calculate_classification_metrics(
                y_true_t2, y_pred_t2, y_prob_t2, prefix="Task2_"
            )
            metrics.update(task2_metrics)

            # Store Task 2 predictions
            if self.per_item:
                predictions["task2"]["lesion_ids"] = df_t2.lesion_id.tolist()
                predictions["task2"]["true_values"] = y_true_t2.tolist()
                predictions["task2"]["predicted_values"] = y_pred_t2.tolist()
                predictions["task2"]["probabilities"] = y_prob_t2.tolist()
                predictions["task2"]["cate_values"] = np.array(all_cates).flatten().tolist()

            logger.info(f"  Task 2 Balanced Accuracy: {task2_metrics['Task2_Balanced_Accuracy']:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating MultiTask CNN: {e}")

        return metrics, predictions

    def save_predictions(self, model_name: str, predictions: Dict):
        """Save per-item predictions to CSV."""
        if not predictions or not self.per_item:
            return

        for task_name, task_preds in predictions.items():
            if not task_preds:
                continue

            df = pd.DataFrame(task_preds)
            output_path = self.output_dir / "predictions" / f"{model_name}_{task_name}_predictions.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {model_name} {task_name} predictions to {output_path}")

    def create_performance_comparison_plot(self):
        """Create bar plot comparing model performances."""
        if not self.visualize:
            return

        logger.info("Creating performance comparison plots...")

        # Prepare data for plotting
        model_names = []
        task1_rmse = []
        task2_bacc = []

        for model_name, metrics in self.all_metrics.items():
            model_names.append(model_name)
            task1_rmse.append(metrics.get("Task1_RMSE", np.nan))
            task2_bacc.append(metrics.get("Task2_Balanced_Accuracy", np.nan))

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Task 1 RMSE comparison
        ax1 = axes[0]
        bars1 = ax1.bar(model_names, task1_rmse, color='steelblue')
        ax1.set_ylabel('RMSE (lower is better)')
        ax1.set_title('Task 1: Clinical Score Prediction')
        ax1.set_ylim(0, max([v for v in task1_rmse if not np.isnan(v)]) * 1.2 if any(not np.isnan(v) for v in task1_rmse) else 1)

        # Add value labels on bars
        for bar, value in zip(bars1, task1_rmse):
            if not np.isnan(value):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{value:.3f}', ha='center', va='bottom')

        # Task 2 Balanced Accuracy comparison
        ax2 = axes[1]
        bars2 = ax2.bar(model_names, task2_bacc, color='darkgreen')
        ax2.set_ylabel('Balanced Accuracy (higher is better)')
        ax2.set_title('Task 2: Treatment Response Prediction')
        ax2.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, value in zip(bars2, task2_bacc):
            if not np.isnan(value):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / "visualizations" / "performance_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved performance comparison plot to {output_path}")

    def create_prediction_scatter_plots(self):
        """Create scatter plots of predictions vs true values."""
        if not self.visualize or not self.predictions:
            return

        logger.info("Creating prediction scatter plots...")

        for model_name, model_preds in self.predictions.items():
            if "task1" not in model_preds or not model_preds["task1"]:
                continue

            task1_preds = model_preds["task1"]
            if "true_values" not in task1_preds or "predicted_values" not in task1_preds:
                continue

            y_true = np.array(task1_preds["true_values"])
            y_pred = np.array(task1_preds["predicted_values"])

            # Create scatter plot
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)

            # Add perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

            # Add regression line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            ax.plot(y_true, p(y_true), "g-", alpha=0.8, label=f'Fitted line: y={z[0]:.2f}x+{z[1]:.2f}')

            ax.set_xlabel('True Clinical Score')
            ax.set_ylabel('Predicted Clinical Score')
            ax.set_title(f'{model_name} - Task 1 Predictions')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add R² annotation
            r2 = r2_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            output_path = self.output_dir / "visualizations" / "prediction_scatter" / f"{model_name}_task1_scatter.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved scatter plot to {output_path}")

    def create_error_distribution_plots(self):
        """Create error distribution analysis plots."""
        if not self.visualize or not self.predictions:
            return

        logger.info("Creating error distribution plots...")

        for model_name, model_preds in self.predictions.items():
            if "task1" not in model_preds or not model_preds["task1"]:
                continue

            task1_preds = model_preds["task1"]
            if "errors" not in task1_preds:
                continue

            errors = np.array(task1_preds["errors"])

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Error histogram
            ax1 = axes[0, 0]
            ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(x=0, color='r', linestyle='--', label='Zero error')
            ax1.set_xlabel('Prediction Error')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Error Distribution')
            ax1.legend()

            # QQ plot
            ax2 = axes[0, 1]
            from scipy import stats
            stats.probplot(errors, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normal Distribution)')

            # Error vs predicted value
            ax3 = axes[1, 0]
            if "predicted_values" in task1_preds:
                y_pred = np.array(task1_preds["predicted_values"])
                ax3.scatter(y_pred, errors, alpha=0.5, s=20)
                ax3.axhline(y=0, color='r', linestyle='--')
                ax3.set_xlabel('Predicted Value')
                ax3.set_ylabel('Prediction Error')
                ax3.set_title('Residual Plot')
                ax3.grid(True, alpha=0.3)

            # Error statistics box
            ax4 = axes[1, 1]
            ax4.axis('off')
            stats_text = f"""Error Statistics:

Mean Error: {np.mean(errors):.4f}
Std Error: {np.std(errors):.4f}
Median Error: {np.median(errors):.4f}
Min Error: {np.min(errors):.4f}
Max Error: {np.max(errors):.4f}

25th Percentile: {np.percentile(errors, 25):.4f}
75th Percentile: {np.percentile(errors, 75):.4f}

Skewness: {stats.skew(errors):.4f}
Kurtosis: {stats.kurtosis(errors):.4f}"""

            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

            plt.suptitle(f'{model_name} - Error Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()

            output_path = self.output_dir / "visualizations" / "error_analysis" / f"{model_name}_error_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved error analysis plot to {output_path}")

    def create_confusion_matrices(self):
        """Create confusion matrix visualizations for classification tasks."""
        if not self.visualize:
            return

        logger.info("Creating confusion matrix plots...")

        for model_name, metrics in self.all_metrics.items():
            cm_key = "Task2_Confusion_Matrix"
            if cm_key not in metrics:
                continue

            cm = np.array(metrics[cm_key])

            # Create confusion matrix plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Non-responder', 'Responder'],
                       yticklabels=['Non-responder', 'Responder'],
                       cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{model_name} - Task 2 Confusion Matrix')

            # Add metrics as text
            bacc = metrics.get("Task2_Balanced_Accuracy", 0)
            precision = metrics.get("Task2_Precision", 0)
            recall = metrics.get("Task2_Recall", 0)
            f1 = metrics.get("Task2_F1_Score", 0)

            metrics_text = f'Balanced Acc: {bacc:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
            ax.text(1.02, 0.5, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='center')

            output_path = self.output_dir / "visualizations" / "confusion_matrices" / f"{model_name}_confusion_matrix.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved confusion matrix to {output_path}")

    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        logger.info("Generating HTML report...")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric-card {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .best-score {{
            font-weight: bold;
            color: #4CAF50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>Comprehensive Model Evaluation Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Test Data:</strong> {self.outcomes_path}</p>
    <p><strong>Number of Test Samples:</strong> {len(self.test_df)}</p>

    <h2>Executive Summary</h2>
    <div class="metric-card">
"""

        # Find best models
        best_task1_model = None
        best_task1_rmse = float('inf')
        best_task2_model = None
        best_task2_bacc = 0

        for model_name, metrics in self.all_metrics.items():
            if "Task1_RMSE" in metrics and metrics["Task1_RMSE"] < best_task1_rmse:
                best_task1_rmse = metrics["Task1_RMSE"]
                best_task1_model = model_name
            if "Task2_Balanced_Accuracy" in metrics and metrics["Task2_Balanced_Accuracy"] > best_task2_bacc:
                best_task2_bacc = metrics["Task2_Balanced_Accuracy"]
                best_task2_model = model_name

        html_content += f"""
        <p><strong>Best Task 1 Model:</strong> {best_task1_model} (RMSE: {best_task1_rmse:.4f})</p>
        <p><strong>Best Task 2 Model:</strong> {best_task2_model} (Balanced Accuracy: {best_task2_bacc:.4f})</p>
    </div>

    <h2>Task 1: Clinical Score Prediction (Regression)</h2>
    <div class="metric-card">
        <table>
            <tr>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>R²</th>
                <th>Pearson r</th>
                <th>Spearman r</th>
            </tr>
"""

        for model_name, metrics in self.all_metrics.items():
            if "Task1_RMSE" in metrics:
                rmse = metrics.get("Task1_RMSE", "-")
                mae = metrics.get("Task1_MAE", "-")
                r2 = metrics.get("Task1_R2", "-")
                pearson = metrics.get("Task1_Pearson_r", "-")
                spearman = metrics.get("Task1_Spearman_r", "-")

                rmse_class = "best-score" if model_name == best_task1_model else ""

                rmse_str = f"{rmse:.4f}" if isinstance(rmse, float) else str(rmse)
                mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
                r2_str = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
                pearson_str = f"{pearson:.4f}" if isinstance(pearson, float) else str(pearson)
                spearman_str = f"{spearman:.4f}" if isinstance(spearman, float) else str(spearman)

                html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td class="{rmse_class}">{rmse_str}</td>
                <td>{mae_str}</td>
                <td>{r2_str}</td>
                <td>{pearson_str}</td>
                <td>{spearman_str}</td>
            </tr>
"""

        html_content += """
        </table>
    </div>

    <h2>Task 2: Treatment Response Prediction (Classification)</h2>
    <div class="metric-card">
        <table>
            <tr>
                <th>Model</th>
                <th>Balanced Accuracy</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
            </tr>
"""

        for model_name, metrics in self.all_metrics.items():
            if "Task2_Balanced_Accuracy" in metrics:
                bacc = metrics.get("Task2_Balanced_Accuracy", "-")
                acc = metrics.get("Task2_Accuracy", "-")
                prec = metrics.get("Task2_Precision", "-")
                rec = metrics.get("Task2_Recall", "-")
                f1 = metrics.get("Task2_F1_Score", "-")
                auc = metrics.get("Task2_ROC_AUC", "-")

                bacc_class = "best-score" if model_name == best_task2_model else ""

                bacc_str = f"{bacc:.4f}" if isinstance(bacc, float) else str(bacc)
                acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
                prec_str = f"{prec:.4f}" if isinstance(prec, float) else str(prec)
                rec_str = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
                f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
                auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)

                html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td class="{bacc_class}">{bacc_str}</td>
                <td>{acc_str}</td>
                <td>{prec_str}</td>
                <td>{rec_str}</td>
                <td>{f1_str}</td>
                <td>{auc_str}</td>
            </tr>
"""

        html_content += """
        </table>
    </div>

    <h2>Visualizations</h2>
    <div class="metric-card">
"""

        # Add visualization images if they exist
        viz_dir = self.output_dir / "visualizations"
        if (viz_dir / "performance_comparison.png").exists():
            html_content += """
        <h3>Performance Comparison</h3>
        <img src="visualizations/performance_comparison.png" alt="Performance Comparison">
"""

        # Add scatter plots
        scatter_dir = viz_dir / "prediction_scatter"
        if scatter_dir.exists():
            scatter_files = list(scatter_dir.glob("*.png"))
            if scatter_files:
                html_content += "<h3>Prediction Scatter Plots</h3>"
                for scatter_file in scatter_files:
                    rel_path = scatter_file.relative_to(self.output_dir)
                    html_content += f'<img src="{rel_path}" alt="Scatter Plot">\n'

        html_content += """
    </div>

    <h2>Output Files</h2>
    <div class="metric-card">
        <ul>
            <li><strong>Metrics:</strong> metrics/summary_metrics.csv, metrics/detailed_metrics.json</li>
            <li><strong>Predictions:</strong> predictions/[model]_[task]_predictions.csv</li>
            <li><strong>Visualizations:</strong> visualizations/</li>
            <li><strong>Logs:</strong> logs/evaluation.log</li>
        </ul>
    </div>

</body>
</html>
"""

        report_path = self.output_dir / "reports" / "evaluation_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Saved HTML report to {report_path}")

    def run(self):
        """Run the complete evaluation pipeline."""
        logger.info("="*60)
        logger.info("Starting Comprehensive Evaluation Pipeline")
        logger.info("="*60)

        # Evaluate each model type
        for model_type in self.models_to_eval:
            if model_type == "baseline":
                metrics, preds = self.evaluate_baseline_models()
                self.all_metrics["Baseline"] = metrics
                self.predictions["Baseline"] = preds
                self.save_predictions("baseline", preds)

            elif model_type == "atlas":
                metrics, preds = self.evaluate_atlas_models()
                self.all_metrics["Atlas"] = metrics
                self.predictions["Atlas"] = preds
                self.save_predictions("atlas", preds)

            elif model_type == "cnn":
                metrics, preds = self.evaluate_cnn_models()
                self.all_metrics["CNN"] = metrics
                self.predictions["CNN"] = preds
                self.save_predictions("cnn", preds)

            elif model_type == "multitask":
                metrics, preds = self.evaluate_multitask_cnn()
                self.all_metrics["MultiTask-CNN"] = metrics
                self.predictions["MultiTask-CNN"] = preds
                self.save_predictions("multitask", preds)

        # Save summary metrics
        self.save_summary_metrics()

        # Generate visualizations
        if self.visualize:
            self.create_performance_comparison_plot()
            self.create_prediction_scatter_plots()
            self.create_error_distribution_plots()
            self.create_confusion_matrices()

        # Generate HTML report
        self.generate_html_report()

        logger.info("="*60)
        logger.info("Evaluation Pipeline Completed Successfully!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)

    def save_summary_metrics(self):
        """Save summary metrics to CSV and JSON."""
        # Prepare data for CSV
        rows = []
        for model_name, metrics in self.all_metrics.items():
            row = {"Model": model_name}
            for key, value in metrics.items():
                if not isinstance(value, (list, dict)):  # Skip complex types for CSV
                    row[key] = value
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.output_dir / "metrics" / "summary_metrics.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved summary metrics to {csv_path}")

        # Save detailed metrics as JSON
        json_path = self.output_dir / "metrics" / "detailed_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self.all_metrics, f, indent=2, default=str)
        logger.info(f"Saved detailed metrics to {json_path}")

        # Create model comparison table
        if rows:
            comparison_cols = [
                "Model", "Task1_RMSE", "Task1_R2",
                "Task2_Balanced_Accuracy", "Task2_F1_Score"
            ]
            available_cols = [col for col in comparison_cols if col in df.columns]
            if len(available_cols) > 1:
                comparison_df = df[available_cols]
                comparison_path = self.output_dir / "metrics" / "model_comparison.csv"
                comparison_df.to_csv(comparison_path, index=False)
                logger.info(f"Saved model comparison to {comparison_path}")


def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation pipeline for brain lesion analysis models"
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=PROCESSED_DATA_DIR / "test.csv",
        help="Path to CSV file with ground truth outcomes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation outputs (default: results/evaluation_TIMESTAMP)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models to evaluate (baseline,atlas,cnn,multitask) or 'all'",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization generation",
    )
    parser.add_argument(
        "--no-per-item",
        action="store_true",
        help="Disable per-item prediction outputs",
    )

    args = parser.parse_args()

    # Parse models to evaluate
    if args.models.lower() == "all":
        models_to_eval = ["baseline", "atlas", "cnn", "multitask"]
    else:
        models_to_eval = [m.strip().lower() for m in args.models.split(",")]

    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"evaluation_{timestamp}"
    else:
        output_dir = args.output_dir

    # Create evaluator and run
    evaluator = ComprehensiveEvaluator(
        outcomes_path=args.outcomes,
        output_dir=output_dir,
        models_to_eval=models_to_eval,
        visualize=not args.no_visualize,
        per_item=not args.no_per_item,
    )

    evaluator.run()


if __name__ == "__main__":
    main()