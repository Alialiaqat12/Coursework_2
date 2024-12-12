from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def train_and_visualize(df, evaluator=None):
    """
    Train a Random Forest model and visualize Heart Rate vs. Activity Type.

    Args:
        df (pd.DataFrame): The input dataframe containing 'Heart Rate' and 'Activity Type'.
        evaluator (ModelEvaluator): An instance of ModelEvaluator to log and compare metrics.
    """
    print("Starting training and visualization...")

    # Extract features (Heart Rate) and labels (Activity Type)
    X = df[["Heart Rate"]]
    y = df["Activity Type"]

    # Debug: Print data samples
    print("Sample Heart Rate data:", X.head())
    print("Sample Activity Type data:", y.head())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    if evaluator:
        evaluator.evaluate("Random Forest", y_test, y_pred)

    # Print metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Visualize Heart Rate vs. Activity Type (Training Data)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train["Heart Rate"], y_train, alpha=0.5, c=y_train, cmap="viridis", label="Training Data")
    plt.colorbar(label="Activity Type")
    plt.xlabel("Heart Rate")
    plt.ylabel("Activity Type")
    plt.title("Heart Rate vs Activity Type (Training Data)")
    plt.legend()
    plt.grid()
    plt.show()
