import os
import pandas as pd
import torch
from transformers import (
    AlbertTokenizerFast,
    AlbertForSequenceClassification,
    TextClassificationPipeline,
)

# 1. Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "albert_finetuned_three_class")
TEST_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_test_labeled.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "..", "predictions", "albert_predictions.csv")

# 2. Load tokenizer & model (3-way)
tokenizer = AlbertTokenizerFast.from_pretrained(MODEL_DIR)
model = AlbertForSequenceClassification.from_pretrained(MODEL_DIR)

# 3. Build a 3-class pipeline
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=128,
    return_all_scores=True,  # returns list of scores for LABEL_0, LABEL_1, LABEL_2
)

# 4. Read test CSV
df_test = pd.read_csv(TEST_CSV).dropna(subset=["comment"])

# 5. Inference in batches
batch_size = 32
all_preds = []

for i in range(0, len(df_test), batch_size):
    texts = df_test["comment"].iloc[i : i + batch_size].tolist()
    batch_scores = pipeline(texts)
    # batch_scores is a list (len=batch) of lists of dicts:
    #   [ [{"label":"LABEL_0","score":0.x}, {"label":"LABEL_1","score":0.y}, {"label":"LABEL_2","score":0.z}], ... ]
    for scores in batch_scores:
        # pick the dict with the highest score
        best = max(scores, key=lambda x: x["score"])
        # map LABEL_0 → 0, LABEL_1 → 1, LABEL_2 → 2
        label_int = int(best["label"].split("_")[1])
        all_preds.append(label_int)

# 6. Attach and save only numeric labels
df_test["predicted_label"] = all_preds
df_test[["comment", "label", "predicted_label"]].to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")
