import torch
from pathlib import Path
from search import search
from transformers import AutoTokenizer, AutoModelForCausalLM

CURRENT_DIR = Path(__file__).parent
MODEL_DIR = CURRENT_DIR / "gpt2-recipes-finetuned"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
model.eval()


# ============================================================
# BASE GENERATOR
# ============================================================


def generate_gpt2(prompt, max_new_tokens=80, temperature=0.8, top_p=0.92):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.12,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ============================================================
# HELPERS
# ============================================================


def extract_top_ingredients(df, top_k=10):
    """Collect most common ingredients from df results."""
    from collections import Counter

    all_ings = []
    for lst in df["ingredient_list"]:
        all_ings.extend(lst)
    freq = Counter(all_ings)
    return [ing for ing, _ in freq.most_common(top_k)]


# ============================================================
# STAGE 1 — TITLE GENERATION
# ============================================================


def generate_title(query, top_ings):
    prompt = f"""
Write a creative and appetizing recipe title.

User request: "{query}"
Inspiration ingredients: {", ".join(top_ings)}

Title: """
    out = generate_gpt2(prompt, max_new_tokens=12, temperature=0.7)

    # take only the first line
    title = out.split("\n")[0].replace("Title:", "").strip()
    return title


# ============================================================
# STAGE 2 — INGREDIENT LIST GENERATION
# ============================================================


def generate_ingredients_block(title, top_ings):
    prompt = f"""
Recipe:
Title: {title}
Ingredients:
- """

    text = generate_gpt2(prompt, max_new_tokens=80, temperature=0.85)
    lines = text.split("\n")

    ingredients = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("- "):
            ingredients.append(ln[2:].strip())
        elif ln.startswith("Directions"):
            break
        elif ln == "":
            break

    # simple cleanup
    ingredients = [x for x in ingredients if 2 <= len(x) <= 60]

    return ingredients


# ============================================================
# STAGE 3 — STEP-BY-STEP DIRECTIONS
# ============================================================


def generate_directions(title, ingredients):
    ing_text = "\n".join(f"- {i}" for i in ingredients)

    prompt = f"""
Recipe:
Title: {title}
Ingredients:
{ing_text}

Directions:
1. """

    text = generate_gpt2(prompt, max_new_tokens=150, temperature=0.9)

    # extract steps lines
    steps = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln.startswith(tuple(str(i) + "." for i in range(1, 10))):
            steps.append(ln)
        if len(steps) >= 8:  # limit steps
            break

    # cleanup
    steps = [s for s in steps if len(s) > 3]

    return steps


# ============================================================
# FULL PIPELINE
# ============================================================


def generate_recipe(query: str):
    # 1) Retrieve semantic matches
    results_df = search(query, top_k=3)
    print(results_df)

    if len(results_df) == 0:
        return "No relevant recipes found."

    # 2) Extract top real ingredients
    top_ings = extract_top_ingredients(results_df, top_k=10)

    # 3) Generate structured recipe via 3-stage GPT-2
    title = generate_title(query, top_ings)
    ingredients = generate_ingredients_block(title, top_ings)
    steps = generate_directions(title, ingredients)

    # Assemble final recipe
    final = f"Recipe:\nTitle: {title}\n\nIngredients:\n"
    for i in ingredients:
        final += f"- {i}\n"

    final += "\nDirections:\n"
    for s in steps:
        final += f"{s}\n"

    return final.strip()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print(generate_recipe("recipe with rice for dinner"))
