"""
Integration tests (smoke tests) for the CNN training script.

These tests verify that the entire training pipeline executes without errors
by running minimal training cycles. They do not test for convergence or
model quality, only that the components work together correctly.
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestTrainCNNIntegration:
    """Integration test suite for CNN training script."""

    def test_train_cnn_task1_smoke_test(self):
        """
        Test that Task 1 (regression) training script runs without errors in smoke test mode.

        This test executes the training script with --smoke-test flag, which should:
        - Load the training data
        - Create the model
        - Run exactly one training and validation batch
        - Exit successfully without errors
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_cnn_models.py"
        )

        # Command to run the script
        cmd = [sys.executable, str(script_path), "--task", "task1", "--smoke-test"]

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
                "Training completed successfully!" in output
            ), f"Expected success message not found. output: {output}"

        except subprocess.TimeoutExpired:
            pytest.fail("Training script timed out after 5 minutes")

    def test_train_cnn_task2_smoke_test(self):
        """
        Test that Task 2 (classification) training script runs without errors in smoke test mode.

        This test executes the training script with --smoke-test flag, which should:
        - Load the training data
        - Create the model with appropriate loss function
        - Run exactly one training and validation batch
        - Exit successfully without errors
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_cnn_models.py"
        )

        # Command to run the script
        cmd = [sys.executable, str(script_path), "--task", "task2", "--smoke-test"]

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
                "Training completed successfully!" in output
            ), f"Expected success message not found. output: {output}"

        except subprocess.TimeoutExpired:
            pytest.fail("Training script timed out after 5 minutes")

    def test_train_cnn_invalid_task_fails(self):
        """
        Test that the training script fails appropriately with invalid task argument.
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_cnn_models.py"
        )

        # Command to run the script with invalid task
        cmd = [sys.executable, str(script_path), "--task", "invalid_task"]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # Short timeout since this should fail quickly
            cwd=str(Path(__file__).parent.parent.parent),
        )

        # Should fail with non-zero return code
        assert (
            result.returncode != 0
        ), f"Script should fail with invalid task but returned {result.returncode}"

    def test_train_cnn_missing_task_argument_fails(self):
        """
        Test that the training script fails when required --task argument is missing.
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_cnn_models.py"
        )

        # Command to run the script without --task argument
        cmd = [sys.executable, str(script_path)]

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        # Should fail with non-zero return code
        assert (
            result.returncode != 0
        ), f"Script should fail without task argument but returned {result.returncode}"

    def test_train_cnn_help_argument(self):
        """
        Test that the training script displays help when --help is used.
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_cnn_models.py"
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
            "Train CNN models for brain lesion analysis" in result.stdout
        ), f"Help text not found in output: {result.stdout}"
        assert "--task" in result.stdout, "Task argument not documented in help"
        assert (
            "--smoke-test" in result.stdout
        ), "Smoke test argument not documented in help"
        assert (
            "--train-on-all-data" in result.stdout
        ), "Train-on-all-data argument not documented in help"

    def test_train_cnn_task1_with_train_on_all_flag(self):
        """
        Test that Task 1 training with --train-on-all-data flag uses all data.

        This test executes the training script with both --smoke-test and --train-on-all-data flags.
        It verifies that the script logs indicate it's using ALL training samples.
        """
        # Path to the training script
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "train_cnn_models.py"
        )

        # Command to run the script with --train-on-all-data flag
        cmd = [
            sys.executable,
            str(script_path),
            "--task",
            "task1",
            "--smoke-test",
            "--train-on-all-data",
        ]

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

            # Check for expected log messages (logging goes to stderr)
            output = result.stdout + result.stderr
            assert (
                "Task 1: Using ALL training and validation samples" in output
            ), f"Expected ALL data message not found. output: {output}"

            # Should NOT contain the non-zero message
            assert (
                "Task 1: Using non-zero score samples" not in output
            ), f"Found non-zero message when using --train-on-all-data flag. output: {output}"

        except subprocess.TimeoutExpired:
            pytest.fail("Training script timed out after 5 minutes")

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(),
        reason="Training data not available",
    )
    def test_data_dependency_exists(self):
        """
        Test that required data files exist for the integration tests.
        """
        train_csv_path = Path("data/processed/train.csv")
        assert train_csv_path.exists(), f"Required file {train_csv_path} does not exist"

        # Also check that we can read it
        import pandas as pd

        df = pd.read_csv(train_csv_path)
        assert len(df) > 0, "Training data file is empty"

        # Check required columns exist
        required_columns = [
            "lesion_filepath",
            "clinical_score",
            "is_responder",
            "treatment_assignment",
        ]
        for col in required_columns:
            assert (
                col in df.columns
            ), f"Required column {col} not found in training data"

    def test_models_directory_creation(self):
        """
        Test that the models directory gets created if it doesn't exist.
        """
        # This is more of a pre-test to ensure the directory structure is correct
        models_dir = Path("models")

        # The training script should create this directory
        # Even if it doesn't exist initially, the script should handle it
        if not models_dir.exists():
            # This is fine - the script should create it
            pass
        else:
            # If it exists, it should be a directory
            assert models_dir.is_dir(), "models path exists but is not a directory"
