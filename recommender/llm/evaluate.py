import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ── 1. Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_TEST = os.path.join(SCRIPT_DIR, "data", "cookbook_test_labeled.csv")
PRED_DIR = os.path.join(SCRIPT_DIR, "predictions")

df_true = pd.read_csv(LABELED_TEST)

models = {
    "SVM (CharNgram)": "linear_classifier_predictions.csv",
    "Naive Bayes": "bayes_predictions.csv",
    "DistilBERT": "distilbert_predictions.csv",
    "ALBERT": "albert_predictions.csv",
    "LSTM": "keras_lstm_predictions.csv",
}


def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    """Compute two‐sided (1−α) bootstrap CI for macro-F1."""
    boot_scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        ft, fp = y_true.iloc[idx], y_pred.iloc[idx]
        f1 = precision_recall_fscore_support(ft, fp, average="macro", zero_division=0)[
            2
        ]
        boot_scores.append(f1)
    lower = np.percentile(boot_scores, 100 * (alpha / 2))
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return lower, upper


# ── 2. Evaluate each model ──────────────────────────────────────────────────
summary = []

for name, fname in models.items():
    path = os.path.join(PRED_DIR, fname)
    df_pred = pd.read_csv(path)

    # Merge on comment (or index if guaranteed same order)
    df = df_true.merge(
        df_pred[["comment", "predicted_label"]], on="comment", how="inner"
    )

    y_true = df["label"]
    y_pred = df["predicted_label"]

    # Flat metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Per-class report
    creport = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "neutral", "positive"],
        zero_division=0,
    )

    # Bootstrap CI for macro-F1
    lo, hi = bootstrap_ci(y_true, y_pred, n_boot=1000)

    # Print details
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}])")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nPer-class classification report:")
    print(creport)

    # Collect for summary table
    summary.append(
        {
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "F1_lower": lo,
            "F1_upper": hi,
        }
    )

# ── 3. Summary DataFrame ───────────────────────────────────────────────────
summary_df = pd.DataFrame(summary).set_index("Model")
print("\n=== Summary Table ===")
print(summary_df[["Accuracy", "Precision", "Recall", "F1"]].round(4))
