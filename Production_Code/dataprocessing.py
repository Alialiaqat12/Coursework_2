import pandas as pd
import os

def load_data(filepath):
    """Load the dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print("Loading dataset...")
    df = pd.read_csv(filepath, delimiter=' ', header=None)
    print(f"Original dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def reduce_dataset(df):
    """Reduce dataset size."""
    # Step 1: Downsample the rows (keep every 5th row)
    print("Downsampling rows...")
    df = df.iloc[::5, :]
    
    # Step 2: Select key columns (adjust based on dataset structure)
    # Example: Keep only Activity Type and Heart Rate columns
    print("Selecting relevant columns...")
    df = df[[1, 2]]  # Adjust column indices based on your dataset
    df.columns = ["Activity Type", "Heart Rate"]  # Rename columns for clarity
    
    print(f"Reduced dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def save_dataset(df, output_filepath, compress=False):
    """Save the reduced dataset."""
    print(f"Saving reduced dataset to {output_filepath}...")
    if compress:
        # Save as compressed Gzipped file
        df.to_csv(output_filepath, sep=' ', index=False, header=False, compression="gzip")
    else:
        # Save as regular file
        df.to_csv(output_filepath, sep=' ', index=False, header=False)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    # File paths
    input_filepath = "Dataset/PAMAP2.dat"  # Original file
    output_filepath = "Dataset/PAMAP2_reduced.dat"  # Reduced file

    try:
        # Step 1: Load the dataset
        df = load_data(input_filepath)
        
        # Step 2: Reduce the dataset size
        df_reduced = reduce_dataset(df)
        
        # Step 3: Save the reduced dataset
        save_dataset(df_reduced, output_filepath, compress=False)
        
        # Check the size of the reduced file
        file_size = os.path.getsize(output_filepath) / (1024 * 1024)  # Size in MB
        print(f"Reduced file size: {file_size:.2f} MB")
        
        if file_size > 100:
            print("Warning: File is still larger than 100 MB. Consider further reduction.")
    
    except Exception as e:
        print(f"Error: {e}")
