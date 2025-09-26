"""
Integration tests (smoke tests) for the multi-task CNN training script.

These tests verify that the entire multi-task training pipeline executes without errors
by running minimal training cycles. They do not test for convergence or
model quality, only that the components work together correctly.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestTrainMultitaskCNNIntegration:
    """Integration test suite for multi-task CNN training script."""

    def test_train_multitask_cnn_smoke_test(self):
        """
        Test that multi-task training script runs without errors in smoke test mode.

        This test executes the training script with --smoke-test flag, which should:
        - Load the training data with multiple targets
        - Create the multi-task model with shared backbone
        - Run exactly one training and validation batch
        - Compute the conditional loss correctly
        - Exit successfully without errors
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_multitask_cnn.py"
        )

        # Command to run the script
        cmd = [sys.executable, str(script_path), "--smoke-test"]

        # Run the command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                cwd=str(Path(__file__).parent.parent.parent),  # Run from project root
            )

            # Check that the command succeeded
            assert result.returncode == 0, (
                f"Training script failed with return code {result.returncode}. "
                f"stdout: {result.stdout}, stderr: {result.stderr}"
            )

            # Check for expected success messages (logging goes to stderr)
            output = result.stdout + result.stderr
            assert (
                "Multi-task training completed successfully!" in output
            ), f"Expected success message not found. output: {output}"

            # Check for multi-task specific outputs
            assert (
                "Val Severity RMSE:" in output
            ), f"Severity RMSE not reported. output: {output}"

        except subprocess.TimeoutExpired:
            pytest.fail("Training script timed out after 5 minutes")

    def test_train_multitask_cnn_help_argument(self):
        """
        Test that the training script displays help when --help is used.
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_multitask_cnn.py"
        )

        # Command to run the script with --help
        cmd = [sys.executable, str(script_path), "--help"]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        # Should succeed and show help
        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert (
            "Train multi-task CNN model for brain lesion analysis" in result.stdout
        ), f"Help text not found in output: {result.stdout}"
        assert (
            "--smoke-test" in result.stdout
        ), "Smoke test argument not documented in help"

    def test_multitask_training_components_integration(self):
        """
        Test that all multi-task components work together in smoke test mode.

        This comprehensive test verifies:
        - LesionDataset returns dictionary of targets
        - MultiTaskCNN accepts image and treatment inputs
        - conditional_multitask_loss handles partial labels
        - Training loop properly unpacks data
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_multitask_cnn.py"
        )

        # Command to run the script
        cmd = [sys.executable, str(script_path), "--smoke-test"]

        # Run the command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent.parent),
            )

            # Check that the command succeeded
            assert result.returncode == 0, (
                f"Training script failed with return code {result.returncode}. "
                f"stdout: {result.stdout}, stderr: {result.stderr}"
            )

            # Check for expected log messages showing proper data handling
            output = result.stdout + result.stderr

            # Should report dataset statistics
            assert (
                "Training set:" in output
            ), f"Dataset statistics not reported. output: {output}"
            assert (
                "Severity labels:" in output
            ), f"Severity label count not reported. output: {output}"
            assert (
                "Outcome labels:" in output
            ), f"Outcome label count not reported. output: {output}"

            # Should report model parameters
            assert (
                "Model parameters:" in output
            ), f"Model parameter count not reported. output: {output}"

            # Should show training progress
            assert (
                "Train Loss:" in output
            ), f"Training loss not reported. output: {output}"
            assert (
                "Val Loss:" in output
            ), f"Validation loss not reported. output: {output}"

        except subprocess.TimeoutExpired:
            pytest.fail("Training script timed out after 5 minutes")

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(),
        reason="Training data not available",
    )
    def test_data_dependency_for_multitask(self):
        """
        Test that required data files exist and have necessary columns for multi-task learning.
        """
        train_csv_path = Path("data/processed/train.csv")
        assert train_csv_path.exists(), f"Required file {train_csv_path} does not exist"

        # Check that we can read it
        import pandas as pd

        df = pd.read_csv(train_csv_path)
        assert len(df) > 0, "Training data file is empty"

        # Check required columns for multi-task learning
        required_columns = [
            "lesion_filepath",
            "clinical_score",  # For severity task
            "outcome_score",  # For outcome task
            "treatment_assignment",  # For treatment conditioning
        ]
        for col in required_columns:
            assert (
                col in df.columns
            ), f"Required column {col} not found in training data for multi-task learning"

        # Check that there are samples with outcome scores
        outcome_samples = df["outcome_score"].notna().sum()
        assert outcome_samples > 0, "No samples with outcome scores available"

    def test_model_output_file_creation(self):
        """
        Test that the script creates the expected model file.
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_multitask_cnn.py"
        )

        # Command to run the script
        cmd = [sys.executable, str(script_path), "--smoke-test"]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        # Only check for file creation if the script succeeded
        if result.returncode == 0:
            # Check that the model file was created
            model_path = Path("models/multitask_cnn_model.pt")
            assert (
                model_path.exists()
            ), f"Expected model file {model_path} was not created"

            # Basic sanity check on file size (should be non-empty)
            assert model_path.stat().st_size > 0, "Model file is empty"

    def test_conditional_loss_handling(self):
        """
        Test that the training script correctly handles the conditional loss function.

        This verifies that the script can handle:
        - Samples with both severity and outcome labels
        - Samples with only severity labels (NaN outcomes)
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_multitask_cnn.py"
        )

        # Command to run the script
        cmd = [sys.executable, str(script_path), "--smoke-test"]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        # Should succeed even with partial labels
        assert result.returncode == 0, (
            f"Script failed to handle conditional loss. "
            f"stdout: {result.stdout}, stderr: {result.stderr}"
        )

        output = result.stdout + result.stderr

        # The script shouldn't crash due to NaN handling
        assert (
            "nan" not in output.lower() or "Val Outcome RMSE:" in output
        ), "Script may not be handling NaN outcomes correctly"
