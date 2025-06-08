import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_train_full.csv")
TEST_CSV = os.path.join(SCRIPT_DIR, "..", "data", "cookbook_test_labeled.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "distilbert_three_class")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
LR = 3e-5
BATCH_SIZE = 16
EPOCHS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. LOAD & PREPARE DATA ───────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV).dropna(subset=["comment"])
df["label"] = df["ratings"].apply(lambda r: 0 if r <= 1 else (1 if r <= 3 else 2))

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
test_df = pd.read_csv(TEST_CSV).dropna(subset=["comment"])

# ── 3. COMPUTE CLASS WEIGHTS ───────────────────────────────────────────────
y_train = train_df["label"].values
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)


# ── 4. DATASET DEFINITION ───────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_len=MAX_LEN):
        self.enc = tokenizer(
            comments.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = labels.values if labels is not None else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_ds = SentimentDataset(train_df["comment"], train_df["label"], tokenizer)
val_ds = SentimentDataset(val_df["comment"], val_df["label"], tokenizer)
test_ds = SentimentDataset(test_df["comment"], test_df["label"], tokenizer)


# ── 5. CUSTOM CLASSIFIER HEAD ──────────────────────────────────────────────
class ExtraHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.dim
        num_labels = config.num_labels
        self.pre_classifier = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, features):
        x = F.relu(self.pre_classifier(features))
        x = self.dropout(x)
        return self.classifier(x)


base_model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, output_hidden_states=False
)
base_model.classifier = ExtraHead(base_model.config)
model = base_model.to(device)


# ── 6. WEIGHTED TRAINER ─────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # simple mean reduction: each batch averaged properly under Trainer
        loss_f = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
        loss = loss_f(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── 7. OPTIMIZER & TRAINING ARGS ───────────────────────────────────────────
optimizer = AdamW(
    model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=LR,
    lr_scheduler_type="linear",
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    tokenizer=tokenizer,
)

# ── 8. TRAIN ───────────────────────────────────────────────────────────────
trainer.train()

# ── 9. EVALUATE & SAVE ─────────────────────────────────────────────────────
val_out = trainer.predict(val_ds)
test_out = trainer.predict(test_ds)

print("\nValidation metrics:", compute_metrics(val_out))
print("Test metrics:      ", compute_metrics(test_out))

trainer.save_model(OUTPUT_DIR)
print(f"\nBest model & tokenizer saved to {OUTPUT_DIR}")
