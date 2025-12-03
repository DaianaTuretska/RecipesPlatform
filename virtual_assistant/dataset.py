import re
import pandas as pd
import os
from pathlib import Path
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

CURRENT_DIR = Path(__file__).parent
ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")
RECIPES_PATH = os.path.join(ASSETS_DIR, "recipes_extended.csv")


cuisine_map = {
    "soy": "asian",
    "taco": "mexican",
    "pasta": "italian",
    "basil": "italian",
    "curry": "indian",
    "cinnamon": "american",
}


def clean(t) -> str:
    if pd.isna(t):
        return ""
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def time_to_minutes(s):
    if pd.isna(s):
        return None
    s = str(s).lower()
    hours = re.search(r"(\d+)\s*hr", s)
    mins = re.search(r"(\d+)\s*min", s)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if mins:
        total += int(mins.group(1))
    return total if total > 0 else None


def parse_ingredient_list(text):
    if pd.isna(text):
        return []
    text = text.lower()
    parts = re.split(r",|\n|;", text)
    cleaned = []

    for p in parts:
        p = p.strip()
        p = re.sub(r"\([^)]*\)", "", p)  # remove parentheses
        p = re.sub(r"[^a-z\s]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    return cleaned


def detect_cuisine(ingredients):
    text = " ".join(ingredients)
    for key, cuis in cuisine_map.items():
        if key in text:
            return cuis
    return "unknown"


def prepare_recipe_dataset() -> pd.DataFrame:
    df = pd.read_csv(RECIPES_PATH)

    df["document_text"] = (
        df["recipe_name"].fillna("")
        + " "
        + df["ingredients"].fillna("")
        + " "
        + df["directions"].fillna("")
        + " "
        + df["nutrition"].fillna("")
    ).apply(clean)

    df["prep_minutes"] = df["prep_time"].apply(time_to_minutes)
    df["cook_minutes"] = df["cook_time"].apply(time_to_minutes)
    df["total_minutes"] = df["total_time"].apply(time_to_minutes)

    df = df[df["document_text"].str.len() > 20]

    df["ingredient_list"] = df["ingredients"].apply(parse_ingredient_list)
    df["cuisine_auto"] = df["ingredient_list"].apply(detect_cuisine)
    df["total_minutes"] = df["total_minutes"].fillna(df["total_minutes"].median())
    df["servings"] = pd.to_numeric(df["servings"], errors="coerce").fillna(2)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.1)

    df["ingredients"] = df["ingredients"].fillna("").astype(str)
    df["directions"] = df["directions"].fillna("").astype(str)
    df["recipe_name"] = df["recipe_name"].fillna("").astype(str)
    df["nutrition"] = df["nutrition"].fillna("").astype(str)

    return df


def prepare_train_dataset():
    df = prepare_recipe_dataset()

    df["train_text"] = (
        "recipe: " + df["recipe_name"].fillna("") + "\n"
        "ingredients: " + df["ingredients"].fillna("") + "\n"
        "directions: " + df["directions"].fillna("") + "\n"
    )
    train_texts = df["train_text"].tolist()

    dataset = Dataset.from_dict({"text": train_texts})
    return dataset


# dataset = prepare_train_dataset()
# model_name = "distilbert-base-uncased"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name)


# def tokenize(batch):
#     return tokenizer(
#         batch["text"], truncation=True, padding="max_length", max_length=256
#     )


# tokenized_dataset = dataset.map(tokenize, batched=True)
# tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# training_args = TrainingArguments(
#     output_dir="distilbert_recipe_finetuned",
#     overwrite_output_dir=True,
#     eval_strategy="steps",
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     save_steps=1000,
#     logging_steps=200,
#     learning_rate=5e-5,
# )

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["test"],
#     data_collator=data_collator,
# )

# trainer.train()

# trainer.save_model("distilbert_recipe_finetuned")
# tokenizer.save_pretrained("distilbert_recipe_finetuned")


# from transformers import AutoModel, AutoTokenizer
# import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert_recipe_finetuned")
# model = AutoModel.from_pretrained("distilbert_recipe_finetuned")

# def embed(text):
#     tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         out = model(**tokens).last_hidden_state[:,0,:]
#     return out


# import torch.nn.functional as F

# def similarity(a, b):
#     return F.cosine_similarity(a, b)
