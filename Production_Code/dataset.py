import pandas as pd

# Replace 'PAMAP2_processed.dat' with your file path
input_file = 'Dataset/PAMAP2_processed.dat'
output_file = 'Dataset/PAMAP2_processed.csv'

# Assuming the data is space-separated, update `delimiter` if needed
try:
    data = pd.read_csv(input_file, delimiter='\s+', header=None)  # Adjust delimiter if necessary
    data.columns = ['Activity', 'HeartRate']  # Update column names if known
    data.to_csv(output_file, index=False)
    print(f"File converted successfully to {output_file}")
except Exception as e:
    print(f"Error: {e}")
