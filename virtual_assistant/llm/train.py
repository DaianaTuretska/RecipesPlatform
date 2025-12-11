import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ---------------- CONFIG ----------------

PARENT_DIR = Path(__file__).parent.parent  # adjust if needed
ASSETS_PATH = PARENT_DIR / "assets"

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATASET_FILE = ASSETS_PATH / "rag_tuning_dataset.jsonl"

OUTPUT_DIR = PARENT_DIR / "phi3_lora_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"--- Loading base model: {MODEL_ID} ---")
print(f"Dataset: {DATASET_FILE}")
print(f"Output dir: {OUTPUT_DIR}")

# ---------------- TOKENIZER ----------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True,
)

# Make sure we have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_prompt(ex):
    """
    Turn one JSONL row with:
      - instruction
      - input
      - output
    into a single training text in the same format used at inference.
    """
    return f"""### System:
{ex["instruction"]}

### User:
{ex["input"]}

### Assistant:
{ex["output"]}"""


def tokenize(example):
    text = build_prompt(example)

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",  # fixed length for stability
        max_length=512,  # you can bump to 768 if VRAM allows
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # For causal LM, labels are just shifted input_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),
    }


# ---------------- DATASET ----------------

raw_dataset = load_dataset(
    "json",
    data_files=str(DATASET_FILE),
)["train"]

print(f"Raw dataset size: {len(raw_dataset)}")

tokenized_dataset = raw_dataset.map(
    tokenize,
    batched=False,
)

# Keep ONLY tensor columns – avoid all the earlier padding errors
tokenized_dataset = tokenized_dataset.select_columns(
    ["input_ids", "attention_mask", "labels"]
)

print("Final dataset columns:", tokenized_dataset.column_names)


# ---------------- BASE MODEL + LORA ----------------

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    attn_implementation="eager",  # no SDPA / flash-attn on Windows
    device_map="auto",
    trust_remote_code=True,
)

# Gradient checkpointing = big VRAM savings
base_model.gradient_checkpointing_enable()
base_model.config.use_cache = False

# Optional but can help on RTX: allow TF32 (safe on Ampere+)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# LoRA config – typical setup for decoder-only models
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # works for Phi-3
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # just to see % of trainable params


# ---------------- DATALOADER COLLATOR ----------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
)


# ---------------- TRAINING ARGS ----------------

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=20,
    # Quick initial run – you can later switch to num_train_epochs
    max_steps=400,
    learning_rate=2e-4,  # LoRA likes higher LR
    fp16=False,  # model weights already fp16; avoid AMP scaler issues
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="adamw_torch_fused",
    group_by_length=True,
    remove_unused_columns=False,  # important with custom forward + LoRA
    report_to="none",  # no WandB, etc.
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    trainer.train()

    # 🔴 IMPORTANT: merge LoRA into base weights before saving
    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()

    print("Saving merged model + tokenizer...")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Finished! Merged model saved to: {OUTPUT_DIR}")
