import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import os
import pickle

# 1. Load & map to three sentiment classes
CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cookbook_train_full.csv"
)
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["comment"])


def map_to_three(r):
    if r <= 1:
        return 0  # negative
    elif r <= 3:
        return 1  # neutral
    else:
        return 2  # positive


df["label"] = df["ratings"].apply(map_to_three)
print("Class distribution:\n", df["label"].value_counts(), "\n")

# 2. Train/validation split (stratified)
train_df, val_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)

# 3. Build Pipeline
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=15000,
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
            ),
        ),
        (
            "clf",
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=1e-4,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

# 4. Train
pipeline.fit(train_df["comment"], train_df["label"])

# 5. Evaluate
val_preds = pipeline.predict(val_df["comment"])
print("Validation Accuracy:", accuracy_score(val_df["label"], val_preds))
print(
    "\nClassification Report:\n",
    classification_report(
        val_df["label"], val_preds, target_names=["negative", "neutral", "positive"]
    ),
)

# 6. Save artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as vf:
    pickle.dump(pipeline.named_steps["tfidf"], vf)
with open(os.path.join(MODEL_DIR, "sgd_classifier.pkl"), "wb") as mf:
    pickle.dump(pipeline.named_steps["clf"], mf)

print(f"Saved vectorizer to {os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')}")
print(f"Saved classifier to {os.path.join(MODEL_DIR, 'sgd_classifier.pkl')}")
