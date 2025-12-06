import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from virtual_assistant.dataset import prepare_train_dataset


PARENT_DIR = Path(__file__).parent.parent
MODEL_PATH = os.path.join(PARENT_DIR, "distilbert_recipe_finetuned")
dataset = prepare_train_dataset()
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=256
    )


tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

training_args = TrainingArguments(
    output_dir="distilbert_recipe_finetuned",
    overwrite_output_dir=True,
    eval_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=200,
    learning_rate=5e-5,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)


if __name__ == "__main__":
    trainer.train()

    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained("distilbert_recipe_finetuned")
