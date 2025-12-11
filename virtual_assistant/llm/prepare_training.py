import os
import pandas as pd
import json
import random
from pathlib import Path

# 1. Configuration
PARENT_DIR = Path(__file__).parent.parent
ASSETS_PATH = os.path.join(PARENT_DIR, "assets")
INPUT_FILE = os.path.join(ASSETS_PATH, "recipes_extended.csv")
OUTPUT_FILE = os.path.join(ASSETS_PATH, "rag_tuning_dataset.jsonl")

# Templates to make the model robust to different user phrasing
QUESTION_TEMPLATES = [
    "How do I cook {name}?",
    "Give me the recipe for {name}.",
    "I want to make {name}. What are the steps?",
    "Can you help me prepare {name}?",
    "Recipe for {name} please.",
]


def clean_text(text):
    """Simple helper to remove floating point errors or empty strings"""
    if pd.isna(text):
        return ""
    return str(text).strip()


def generate_jsonl():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(
            f"Error: Could not find {INPUT_FILE}. Make sure it is in the same folder."
        )
        return

    # Filter out bad rows (recipes with no name or instructions are useless)
    initial_count = len(df)
    df = df.dropna(subset=["recipe_name", "ingredients", "directions"])
    print(f"Filtered {initial_count} rows down to {len(df)} valid recipes.")

    dataset_entries = []

    for _, row in df.iterrows():
        recipe_name = clean_text(row["recipe_name"])
        ingredients = clean_text(row["ingredients"])
        directions = clean_text(row["directions"])

        # --- A. SIMULATE THE RETRIEVAL (The "Context") ---
        # This represents what your DistilBERT engine finds and gives to the LLM.
        # We simulate a "database dump" format here.
        context_block = f"""
Title: {recipe_name}
Ingredients List: {ingredients}
Method: {directions}
"""

        # --- B. SIMULATE THE USER (The "Question") ---
        user_query = random.choice(QUESTION_TEMPLATES).format(name=recipe_name)

        # --- C. DEFINE THE IDEAL BEHAVIOR (The "Output") ---
        # We teach the LLM to take the raw "context_block" and format it beautifully.
        ideal_response = f"""Here is the recipe for {recipe_name}!

**Ingredients:**
{ingredients}

**Instructions:**
{directions}

Enjoy cooking!"""

        # --- D. FORMAT FOR TRAINING (Alpaca / Llama 3 Format) ---
        entry = {
            "instruction": "You are a specialized Chef Assistant. Use the provided Context to answer the user's question accurately.",
            "input": f"Context:\n{context_block}\n\nQuestion:\n{user_query}",
            "output": ideal_response,
        }

        dataset_entries.append(entry)

    # Save to JSONL
    print(f"Saving {len(dataset_entries)} examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dataset_entries:
            json.dump(entry, f)
            f.write("\n")

    print("Done! You can now use this file with Unsloth.")


if __name__ == "__main__":
    generate_jsonl()
