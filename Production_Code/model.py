import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

def train_and_visualize(df):
    """Train a model and visualize results."""
    print("Training model...")

    # Extract features (Heart Rate) and labels (Activity Type)
    X = df[["Heart Rate"]]
    y = df["Activity Type"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Visualize Heart Rate vs. Activity Type
    plt.figure(figsize=(10, 6))
    plt.scatter(X["Heart Rate"], y, alpha=0.5, label="Actual Activities", color="blue")
    plt.xlabel("Heart Rate")
    plt.ylabel("Activity Type")
    plt.title("Heart Rate vs Activity Type")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Replace with your dataset path
    from dataprocessing import load_and_preprocess_data

    filepath = "Dataset/PAMAP2.dat"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    # Load and preprocess the data
    df = load_and_preprocess_data(filepath)

    # Train the model and visualize
    train_and_visualize(df)
