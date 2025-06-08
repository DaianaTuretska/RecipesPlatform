import os
import glob
import torch
import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TextClassificationPipeline,
)
from safetensors.torch import load_file as load_safetensors

# 1) Locate your latest checkpoint dir
root = os.path.join(os.path.dirname(__file__), "distilbert_three_class")
ckpts = glob.glob(os.path.join(root, "checkpoint-*"))
latest = max(ckpts, key=lambda d: int(d.rsplit("-", 1)[-1]))
print("Loading from", latest)

# 2) Load the model directly (it will pick up the saved default head)
model = DistilBertForSequenceClassification.from_pretrained(
    latest, from_tf=False, use_safetensors=True  # tells HF to use model.safetensors
)

# 3) Load safetensors weights (HuggingFace will wire them into the default head)
state = load_safetensors(os.path.join(latest, "model.safetensors"), device="cpu")
model.load_state_dict(state)

# 4) Build your pipeline
tokenizer = DistilBertTokenizerFast.from_pretrained(latest)
device = 0 if torch.cuda.is_available() else -1
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=device,
    truncation=True,
    max_length=128,
)

# 5) Run inference
df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "data", "cookbook_test_labeled.csv")
).dropna(subset=["comment"])
preds = pipe(df["comment"].tolist(), batch_size=32)

# 6) Map labels & save
hf_to_int = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
sent_map = {0: "negative", 1: "neutral", 2: "positive"}
df["predicted_label"] = [hf_to_int[p["label"]] for p in preds]
df["predicted_sentiment"] = df["predicted_label"].map(sent_map)

out = os.path.join(
    os.path.dirname(__file__), "..", "predictions", "distilbert_predictions.csv"
)
os.makedirs(os.path.dirname(out), exist_ok=True)
df[["comment", "label", "predicted_label", "predicted_sentiment"]].to_csv(
    out, index=False
)
print("Saved to", out)
