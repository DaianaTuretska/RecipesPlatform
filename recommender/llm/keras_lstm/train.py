import os
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_train_full.csv")
TOKENIZER_FP = os.path.join(SCRIPT_DIR, "model", "lstm_tokenizer.pkl")
MODEL_FP = os.path.join(SCRIPT_DIR, "model", "lstm_three_class_balanced.h5")

VOCAB_SIZE = 20000
MAX_LEN = 100
EMBED_DIM = 100
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

# ── 1. LOAD & LABEL DATA ────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV).dropna(subset=["comment"])


def map_to_three(r):
    if r <= 1:
        return 0  # negative
    if r <= 3:
        return 1  # neutral
    return 2  # positive


df["label"] = df["ratings"].apply(map_to_three)
print("Original class counts:", df["label"].value_counts().to_dict())

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# ── 2. UPSAMPLE TRAINING CLASSES ────────────────────────────────────────────────
# Separate each class
pos = train_df[train_df.label == 2]
neg = train_df[train_df.label == 0]
neu = train_df[train_df.label == 1]

# Target count = majority class size
target_n = len(pos)

neg_up = neg.sample(target_n, replace=True, random_state=42)
neu_up = neu.sample(target_n, replace=True, random_state=42)

train_balanced = pd.concat([pos, neg_up, neu_up]).sample(frac=1, random_state=42)
print("Balanced class counts:", train_balanced["label"].value_counts().to_dict())

# ── 3. TOKENIZATION ─────────────────────────────────────────────────────────────
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(train_balanced["comment"])


def make_padded_sequences(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")


X_train = make_padded_sequences(train_balanced["comment"])
y_train = train_balanced["label"].values

X_val = make_padded_sequences(val_df["comment"])
y_val = val_df["label"].values

# ── 4. BUILD & COMPILE MODEL ───────────────────────────────────────────────────
model = Sequential(
    [
        Embedding(VOCAB_SIZE, EMBED_DIM, input_shape=(MAX_LEN,)),
        LSTM(LSTM_UNITS),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(3, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(LEARNING_RATE),
    metrics=["accuracy"],
)
model.summary()

# ── 5. TRAIN ────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# ── 6. VALIDATION DIAGNOSTICS ───────────────────────────────────────────────────
val_probs = model.predict(X_val, batch_size=BATCH_SIZE)
val_preds = val_probs.argmax(axis=1)

print("Predicted counts:", np.bincount(val_preds, minlength=3))
print("True      counts:", np.bincount(y_val, minlength=3))

for cls in [0, 1, 2]:
    mask = y_val == cls
    acc = (val_preds[mask] == cls).mean()
    print(f"Class {cls} accuracy: {acc:.3f}")

cm = confusion_matrix(y_val, val_preds, labels=[0, 1, 2])
print("\nConfusion Matrix:\n", cm)

# Plot
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(cm, cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=[0, 1, 2],
    yticks=[0, 1, 2],
    xticklabels=["neg", "neu", "pos"],
    yticklabels=["neg", "neu", "pos"],
    xlabel="Predicted",
    ylabel="True",
    title="Confusion Matrix",
)
th = cm.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(
            j, i, cm[i, j], ha="center", color="white" if cm[i, j] > th else "black"
        )
plt.tight_layout()
plt.show()

print(
    "\n3-Class Report:\n",
    classification_report(y_val, val_preds, target_names=["neg", "neu", "pos"]),
)

# ── 7. SAVE ARTIFACTS ──────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_FP), exist_ok=True)
model.save(MODEL_FP)
with open(TOKENIZER_FP, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"Saved model to   {MODEL_FP}")
print(f"Saved tokenizer to {TOKENIZER_FP}")
