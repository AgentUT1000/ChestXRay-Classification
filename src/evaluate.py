import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, multilabel_confusion_matrix, classification_report
)
import seaborn as sns

def evaluate_model(model, X_test, y_test, class_names, threshold=0.5):
    """
    Evaluate a trained model on test data and generate metrics + visualizations.

    Args:
        model (tf.keras.Model): Trained Keras model.
        X_test (np.ndarray): Test images.
        y_test (np.ndarray): Ground truth labels (binary multi-label format).
        class_names (list): List of label names.
        threshold (float): Threshold for converting probabilities to binary.

    Returns:
        dict: Dictionary of evaluation scores.
    """
    os.makedirs("outputs/plots", exist_ok=True)

    # Predict probabilities
    y_prob = model.predict(X_test, verbose=1)

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Compute evaluation scores
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_test, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
    }

    # Classification Report (text)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix (one per class)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    for i, cm in enumerate(mcm):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {class_names[i]}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/confusion_{class_names[i]}.png")
        plt.close()

    # ROC Curve for each class
    for i, label in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/plots/roc_{label}.png")
        plt.close()

    return results
