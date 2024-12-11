import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(filepath):
    """Load the dataset and assign column names."""
    df = pd.read_csv(filepath, delimiter=' ', header=None)

    # Dynamically generate column names based on the number of columns in the dataset
    column_count = df.shape[1]
    column_names = [f"Feature_{i+1}" for i in range(column_count)]
    column_names[1] = "Activity Type"  # Assign meaningful name to Activity Type
    column_names[2] = "Heart Rate"     # Assign meaningful name to Heart Rate
    df.columns = column_names
    return df

def preprocess_data(df):
    """Preprocess the dataset."""
    # Fill missing values
    df["Heart Rate"].fillna(method="ffill", inplace=True)

    # Filter data if needed (e.g., remove invalid activity types)
    df = df[df["Activity Type"].notna()]

    # Select features and labels
    features = df[["Heart Rate"]]  # Heart rate column
    labels = df["Activity Type"]  # Activity type column

    return features, labels

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Replace with the correct file path
    filepath = "Dataset/PAMAP2.dat"

    # Load and preprocess the dataset
    df = load_data(filepath)
    features, labels = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
