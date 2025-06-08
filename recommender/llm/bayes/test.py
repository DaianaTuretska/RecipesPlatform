import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# 1. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE_DIR, "..", "data", "cookbook_test_labeled.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer_bayes.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model", "multinomial_nb_model.pkl")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "predictions", "bayes_predictions.csv")

# 2. Load artifacts
with open(VECTORIZER_PATH, "rb") as vf:
    vectorizer = pickle.load(vf)
with open(MODEL_PATH, "rb") as mf:
    clf = pickle.load(mf)

# 3. Load & preprocess test data
df_test = pd.read_csv(TEST_CSV)
df_test = df_test.dropna(subset=["comment"])

# 4. Inference
X_test = vectorizer.transform(df_test["comment"])
df_test["predicted_label"] = clf.predict(X_test)

# 5. Map numeric labels → human-readable
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
df_test["predicted_sentiment"] = df_test["predicted_label"].map(sentiment_map)


# 6. Optional evaluation if ratings are present
def map_to_three(r):
    if r <= 1:
        return 0
    elif r <= 3:
        return 1
    else:
        return 2


if "ratings" in df_test.columns:
    df_test["true_label"] = df_test["ratings"].apply(map_to_three)
    df_test["true_sentiment"] = df_test["true_label"].map(sentiment_map)

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

if "predicted_sentiment" in df_test.columns:
    df_test = df_test.drop(columns=["predicted_sentiment"])

# 8. Save only comment, ratings (if any), and predicted_sentiment
df_test.to_csv(OUTPUT_CSV, index=False)
print(f"Saved test predictions to:\n  {OUTPUT_CSV}")
