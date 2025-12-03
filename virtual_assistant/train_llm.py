# train_gpt2_recipes.py
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

CURRENT_DIR = Path(__file__).parent
DATA_PATH = CURRENT_DIR / "assets" / "recipes_for_lm.txt"
OUTPUT_DIR = CURRENT_DIR / "gpt2-recipes-finetuned"

MODEL_NAME = "gpt2"  # or "distilgpt2" if your GPU is small

# 1. Load raw text as a dataset
dataset = load_dataset("text", data_files={"train": str(DATA_PATH)}, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# GPT-2 has no pad token by default
tokenizer.pad_token = tokenizer.eos_token


def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
    )


tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"],
)

# Remove empty input_ids
tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id
model.gradient_checkpointing_enable()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    num_train_epochs=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_steps=2000,
    logging_steps=100,
    bf16=True,  # if GPU supports
    fp16=False,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="epoch",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)


trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
