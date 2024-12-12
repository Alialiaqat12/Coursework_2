# dataprocessing.py
import pandas as pd
import os

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, delimiter=' ', header=None)

    # Rename columns based on dataset structure
    if df.shape[1] == 2:
        df.columns = ["Activity Type", "Heart Rate"]
    else:
        raise ValueError("Unexpected dataset structure. Expected 2 columns: 'Activity Type' and 'Heart Rate'.")

    # Handle missing values in Heart Rate
    df["Heart Rate"].fillna(method="ffill", inplace=True)  # Forward fill
    df["Heart Rate"].fillna(df["Heart Rate"].mean(), inplace=True)  # Fill remaining NaN with mean

    print(f"Dataset loaded and preprocessed. Shape: {df.shape}")

    # Save processed dataset
    processed_filepath = "Dataset/PAMAP2_processed.dat"
    df.to_csv(processed_filepath, sep=' ', index=False, header=False)
    print(f"Processed dataset saved to: {processed_filepath}")
    return df

if __name__ == "__main__":
    filepath = "Dataset/PAMAP2.dat"
    df = load_and_preprocess_data(filepath)
    print("Sample Data:\n", df.head())
