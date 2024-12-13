from dataprocessing import load_and_preprocess_data
from ModelEvaluator import ModelEvaluator
from modeltrainer import train_and_visualize
import  pandas as pd
import os
import argparse
import mlflow

def main():
    print("Starting the pipeline...")

    # Get the arugments we need to avoid fixing the dataset path in code
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
    args = parser.parse_args()
    mlflow.autolog()

    # Specify the dataset path
    # filepath = "Dataset/PAMAP2_processed.dat"

    df = pd.read_csv(args.trainingdata, delimiter=' ', header=None)
    # df = pd.read_csv(filepath, delimiter=' ', header=None)

    # Rename columns based on dataset structure
    if df.shape[1] == 2:
        df.columns = ["Activity Type", "Heart Rate"]
    else:
        raise ValueError("Unexpected dataset structure. Expected 2 columns: 'Activity Type' and 'Heart Rate'.")

    # Handle missing values in Heart Rate
    df["Heart Rate"].fillna(method="ffill", inplace=True)  # Forward fill
    df["Heart Rate"].fillna(df["Heart Rate"].mean(), inplace=True)  # Fill remaining NaN with mean

    print(df)

    # Check if the processed dataset exists
    # if not os.path.exists(filepath):
    #     print("Preprocessing raw dataset...")
    #     raw_filepath = "Dataset/PAMAP2.dat"
    #     df = load_and_preprocess_data(raw_filepath)
    # else:
    #     print(f"Processed dataset already exists at: {filepath}")
    #     df = load_and_preprocess_data(filepath)

    print(f"Dataset shape: {df.shape}")

    # Initialize the evaluator
    evaluator = ModelEvaluator()

    # Train the model and visualize results
    print("Calling train_and_visualize...")
    train_and_visualize(df, evaluator)

    # Compare model performance
    evaluator.compare_performance()

    # Check for performance regression
    if not evaluator.check_performance_regression(threshold=0.55):
        print("Performance regression detected. Investigate further.")
    else:
        print("No performance regression detected.")

if __name__ == "__main__":
    main()
