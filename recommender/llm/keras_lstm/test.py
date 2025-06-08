import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_test_labeled.csv")
MODEL_FP = os.path.join(SCRIPT_DIR, "model", "lstm_three_class.h5")
TOKENIZER_FP = os.path.join(SCRIPT_DIR, "model", "lstm_tokenizer.pkl")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "..", "predictions", "keras_lstm_predictions.csv")
MAX_LEN = 100
BATCH_SIZE = 64

# ── 1. LOAD MODEL & TOKENIZER ───────────────────────────────────────────────────
model = load_model(MODEL_FP)

print(model.summary())

with open(TOKENIZER_FP, "rb") as f:
    tokenizer = pickle.load(f)

# ── 2. READ TEST DATA ───────────────────────────────────────────────────────────
df_test = pd.read_csv(TEST_CSV).dropna(subset=["comment"])
seqs = tokenizer.texts_to_sequences(df_test["comment"])
X_test = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

# ── 3. PREDICT ─────────────────────────────────────────────────────────────────
# This returns an (n_samples, 3) array of class probabilities
preds_prob = model.predict(X_test, batch_size=BATCH_SIZE)
print(preds_prob)
# Take argmax to get integer labels 0,1,2
preds_label = np.argmax(preds_prob, axis=1)

# ── 4. SAVE ────────────────────────────────────────────────────────────────────
df_test["predicted_label"] = preds_label
df_test.to_csv(OUTPUT_CSV, index=False)
print(f"Saved LSTM three-class predictions to: {OUTPUT_CSV}")
