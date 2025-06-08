import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AlbertTokenizerFast,
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_train_full.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "albert_finetuned_three_class")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

MODEL_NAME = "albert-base-v2"
MAX_LEN = 128
EPOCHS = 3
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. LOAD & PREP DATA ─────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV).dropna(subset=["comment"])


def map_to_three(r):
    if r <= 1:
        return 0  # negative
    elif r <= 3:
        return 1  # neutral
    else:
        return 2  # positive


df["label"] = df["ratings"].apply(map_to_three)
print("Class distribution:", df["label"].value_counts().to_dict())

train_df, val_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)

# ── 3. TOKENIZER & DATASET ─────────────────────────────────────────────────────
tokenizer = AlbertTokenizerFast.from_pretrained(MODEL_NAME)


class TorchCommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = df["comment"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_ds = TorchCommentDataset(train_df, tokenizer)
val_ds = TorchCommentDataset(val_df, tokenizer)

# ── 4. LOAD MODEL ───────────────────────────────────────────────────────────────
model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(
    DEVICE
)


# ── 5. METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="macro"
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ── 6. TRAINING ARGS ────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100,
    no_cuda=not torch.cuda.is_available(),
)

# ── 7. TRAINER ─────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# ── 8. TRAIN & EVALUATE ─────────────────────────────────────────────────────────
trainer.train()
eval_metrics = trainer.evaluate()
print("ALBERT Validation metrics:", eval_metrics)

# Detailed 3-class report
preds_output = trainer.predict(val_ds)
preds = preds_output.predictions.argmax(-1)
labels = preds_output.label_ids
print("\n3-Class Classification Report:")
print(
    classification_report(
        labels, preds, target_names=["negative", "neutral", "positive"]
    )
)

# ── 9. SAVE MODEL ───────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
print(f"Model, checkpoints, and logs saved to {OUTPUT_DIR}")
