import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "..", "data", "cookbook_train_full.csv")
VECTORIZER_PKL = os.path.join(BASE_DIR, "model", "tfidf_vectorizer_bayes.pkl")
MODEL_PKL = os.path.join(BASE_DIR, "model", "multinomial_nb_model.pkl")

# 1. Load and prepare training data
df = pd.read_csv(TRAIN_CSV).dropna(subset=["comment"])


# Map ratings into 3 classes
def map_to_three(r):
    if r <= 1:
        return 0  # negative
    elif r <= 3:
        return 1  # neutral
    else:
        return 2  # positive


df["label"] = df["ratings"].apply(map_to_three)

# Optional: inspect distribution
print("Class distribution:\n", df["label"].value_counts(), "\n")

# 2. Split into train/validation (stratified)
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Vectorize text with TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words="english",
)
X_train = vectorizer.fit_transform(train_df["comment"])
X_val = vectorizer.transform(val_df["comment"])
y_train = train_df["label"].values
y_val = val_df["label"].values

# 4. Train Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 5. Evaluate on validation set
val_preds = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("\n3-Class Classification Report:")
print(
    classification_report(
        y_val, val_preds, target_names=["negative", "neutral", "positive"]
    )
)

# 6. Save vectorizer and model
os.makedirs(os.path.dirname(VECTORIZER_PKL), exist_ok=True)
with open(VECTORIZER_PKL, "wb") as vf:
    pickle.dump(vectorizer, vf)
with open(MODEL_PKL, "wb") as mf:
    pickle.dump(clf, mf)

print(f"\nSaved vectorizer to: {VECTORIZER_PKL}")
print(f"Saved model to:      {MODEL_PKL}")
