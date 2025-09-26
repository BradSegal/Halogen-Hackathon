# File: scripts/prepare_data.py

import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lesion_analysis.data.loader import load_and_prepare_data

# Define project root relative to the script location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
LESIONS_DIR = DATA_ROOT / "lesions" / "lesions"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"


def main():
    """
    Main function to execute the data preparation and splitting process.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Prepare and split data for training, validation, and testing."
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Process all data without train/val/test splitting (for test suite processing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for processed data (default: data/processed)"
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROCESSED_DATA_DIR

    print("Starting data preparation...")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(
        csv_path=DATA_ROOT / "tasks.csv", lesions_dir=LESIONS_DIR
    )
    print(f"Loaded and validated {len(df)} records.")

    if args.no_split:
        # No-split mode: save all data as a single dataset
        print("Running in no-split mode (test suite processing)")
        full_dataset_path = output_dir / "full_dataset.csv"
        df.to_csv(full_dataset_path, index=False)
        print(f"Full dataset saved to: {full_dataset_path}")
        print("Data preparation complete (no-split mode).")
    else:
        # Normal mode: perform train/val/test split
        # Stratify on treatment_assignment, filling NaNs for stratification to work
        stratify_col = df["treatment_assignment"].fillna("None")

        # Perform a 85/15 split first (to get 15% test set)
        train_val_df, test_df = train_test_split(
            df, test_size=0.15, random_state=42, stratify=stratify_col
        )

        # Split the 85% into train and validation
        # 70% train + 15% val = 85%, so val is 15/85 = 0.1765 of the train_val set
        stratify_col_train_val = train_val_df["treatment_assignment"].fillna("None")
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.1765,  # 0.15 / 0.85 to get 15% of original
            random_state=42,
            stratify=stratify_col_train_val,
        )

        print(
            f"Data split into {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples."
        )

        # Save splits
        train_path = output_dir / "train.csv"
        val_path = output_dir / "validation.csv"
        test_path = output_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train data saved to: {train_path}")
        print(f"Validation data saved to: {val_path}")
        print(f"Test data saved to: {test_path}")
        print("Data preparation complete.")


if __name__ == "__main__":
    main()
