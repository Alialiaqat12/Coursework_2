from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Class for evaluating and logging model performance."""

    def __init__(self):
        self.performance_log = {
            "Model": [],
            "Accuracy": [],
            "Precision (Macro Avg)": [],
            "Recall (Macro Avg)": [],
            "F1-Score (Macro Avg)": []
        }

    def evaluate(self, model_name, y_test, y_pred):
        """Evaluate the model and log metrics."""
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        self.performance_log["Model"].append(model_name)
        self.performance_log["Accuracy"].append(accuracy)
        self.performance_log["Precision (Macro Avg)"].append(report["macro avg"]["precision"])
        self.performance_log["Recall (Macro Avg)"].append(report["macro avg"]["recall"])
        self.performance_log["F1-Score (Macro Avg)"].append(report["macro avg"]["f1-score"])

        # Print evaluation summary
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro Avg): {report['macro avg']['precision']:.4f}")
        print(f"Recall (Macro Avg): {report['macro avg']['recall']:.4f}")
        print(f"F1-Score (Macro Avg): {report['macro avg']['f1-score']:.4f}")
        return accuracy

    def compare_performance(self):
        """Compare performance across all evaluated models."""
        print("\nModel Performance Comparison:")
        for i in range(len(self.performance_log["Model"])):
            print(f"{self.performance_log['Model'][i]} - "
                  f"Accuracy: {self.performance_log['Accuracy'][i]:.4f}, "
                  f"Precision: {self.performance_log['Precision (Macro Avg)'][i]:.4f}, "
                  f"Recall: {self.performance_log['Recall (Macro Avg)'][i]:.4f}, "
                  f"F1-Score: {self.performance_log['F1-Score (Macro Avg)'][i]:.4f}")

    def visualize_confusion_matrix(self, y_test, y_pred, model_name):
        """Visualize confusion matrix."""
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical')
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    def check_performance_regression(self, threshold=0.5):
        """
        Check for performance regressions by ensuring accuracy exceeds a threshold.
        Args:
            threshold (float): Minimum acceptable accuracy.
        Returns:
            bool: True if no regression is detected, False otherwise.
        """
        regression_flag = True
        print("\nChecking for Performance Regression...")
        for i, model in enumerate(self.performance_log["Model"]):
            accuracy = self.performance_log["Accuracy"][i]
            if accuracy < threshold:
                print(f"Performance Regression Detected: {model} has accuracy below threshold ({threshold:.2f})")
                regression_flag = False
        return regression_flag

    def plot_roc_curve(self, model, X_test, y_test, model_name):
        """Plot ROC curve for binary or multi-class classification."""
        if len(y_test.unique()) == 2:  # Binary classification
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            plt.show()
        else:  # Multi-class classification
            # Binarize labels for multi-class ROC curve
            y_test_bin = label_binarize(y_test, classes=list(range(len(y_test.unique()))))
            y_score = model.predict_proba(X_test)

            # Compute ROC curve and ROC area for each class
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Multi-class ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            plt.show()
