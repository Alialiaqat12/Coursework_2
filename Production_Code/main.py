from dataprocessing import load_and_preprocess_data
from ModelEvaluator import ModelEvaluator
from modeltrainer import train_and_visualize
import os

def main():
    print("Starting the pipeline...")

    # Specify the dataset path
    filepath = "Dataset/PAMAP2_processed.dat"

    # Check if the processed dataset exists
    if not os.path.exists(filepath):
        print("Preprocessing raw dataset...")
        raw_filepath = "Dataset/PAMAP2.dat"
        df = load_and_preprocess_data(raw_filepath)
    else:
        print(f"Processed dataset already exists at: {filepath}")
        df = load_and_preprocess_data(filepath)

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
