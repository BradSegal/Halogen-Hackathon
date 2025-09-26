# File: scripts/prepare_data.py

from pathlib import Path
from sklearn.model_selection import train_test_split
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
    print("Starting data preparation...")
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data(
        csv_path=DATA_ROOT / "tasks.csv", lesions_dir=LESIONS_DIR
    )
    print(f"Loaded and validated {len(df)} records.")

    # Stratify on treatment_assignment, filling NaNs for stratification to work
    stratify_col = df["treatment_assignment"].fillna("None")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify_col
    )
    print(f"Data split into {len(train_df)} training and {len(test_df)} test samples.")

    # Save splits
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
