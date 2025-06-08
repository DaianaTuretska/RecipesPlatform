import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report

# 1. Locate & load vectorizer + model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(SCRIPT_DIR, "model/tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(SCRIPT_DIR, "model/sgd_classifier.pkl")

with open(VECTORIZER_PATH, "rb") as vf:
    vectorizer = pickle.load(vf)
with open(MODEL_PATH, "rb") as mf:
    clf = pickle.load(mf)

# 2. Read test CSV & predict
TEST_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_test_labeled.csv")
OUTPUT_CSV = os.path.join(
    SCRIPT_DIR, "..", "predictions", "linear_classifier_predictions.csv"
)

df_test = pd.read_csv(TEST_CSV).dropna(subset=["comment"])
X_test = vectorizer.transform(df_test["comment"])
df_test["predicted_label"] = clf.predict(X_test)


# 3. If you have true ratings, map & evaluate in 3 classes
def map_to_three(r):
    if r <= 1:
        return 0
    elif r <= 3:
        return 1
    else:
        return 2


if "ratings" in df_test.columns:
    df_test["true_label"] = df_test["ratings"].apply(map_to_three)
    y_true = df_test["true_label"].values
    y_pred = df_test["predicted_label"].values

    print("**3-Class Evaluation on Test Set**")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(
        "\nClassification Report:\n",
        classification_report(
            y_true, y_pred, target_names=["negative", "neutral", "positive"]
        ),
    )
else:
    print("No ground-truth 'ratings' column found. Skipping evaluation.")

# 4. Save predictions
df_test.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved predictions to {OUTPUT_CSV}")
