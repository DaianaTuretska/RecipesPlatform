import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
ASSETS = CURRENT_DIR / "assets"

DF_PATH = ASSETS / "recipe_df.pkl"
OUT_PATH = ASSETS / "recipes_for_lm.txt"

df = pd.read_pickle(DF_PATH)


def clean_text(t):
    if not isinstance(t, str):
        return ""
    return t.replace("\n", " ").strip()


def write_recipe(f, title, ingredients, directions):
    f.write("Recipe:\n")
    f.write(f"Title: {title}\n")
    f.write("Ingredients:\n")

    for ing in ingredients:
        ing = clean_text(ing)
        if len(ing) > 1:
            f.write(f"- {ing}\n")

    f.write("Directions:\n")

    # split directions into "sentences"
    steps_raw = clean_text(directions).split(".")
    steps = [s.strip() for s in steps_raw if len(s.strip()) > 3]

    # write max ~6 steps
    for i, step in enumerate(steps[:6], start=1):
        f.write(f"{i}. {step}.\n")

    f.write("END_OF_RECIPE\n\n")


with open(OUT_PATH, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        title = clean_text(row["recipe_name"])
        ingredients = row["ingredient_list"]
        directions = clean_text(row["directions"])
        write_recipe(f, title, ingredients, directions)

print("DONE. Generated clean recipes_for_lm.txt")
