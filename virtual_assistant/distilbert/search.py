import torch.nn.functional as F
import torch
import os
import re
import pandas as pd
from pathlib import Path
from virtual_assistant.distilbert.embed import embed_text

# Paths
PARENT_DIR = Path(__file__).parent.parent
ASSETS_PATH = os.path.join(PARENT_DIR, "assets")

MULTI_EMBEDDINGS_PATH = os.path.join(ASSETS_PATH, "recipe_embeddings_multi.pt")
DATA_FRAME_PATH = os.path.join(ASSETS_PATH, "recipe_df.pkl")
ALL_INGREDIENTS = os.path.join(ASSETS_PATH, "all_ingredients_list.pt")
ALL_INGREDIENT_EMBS = os.path.join(ASSETS_PATH, "ingredient_embs.pt")

df = pd.read_pickle(DATA_FRAME_PATH)

multi = torch.load(MULTI_EMBEDDINGS_PATH, map_location="cpu")
full_emb = multi["full_emb"]  # (N, 768)
ingredients_emb = multi["ingredients_emb"]  # (N, 768)

# Load ingredient vocabulary + embeddings
all_ingredients = torch.load(ALL_INGREDIENTS, map_location="cpu")
ingredient_embs = torch.load(ALL_INGREDIENT_EMBS, map_location="cpu")


# ─────────────────────────────────────────────────────────────
# INGREDIENT + TIME EXTRACTION
# ─────────────────────────────────────────────────────────────

INGREDIENT_TRIGGERS = {
    "with",
    "using",
    "contains",
    "contain",
    "use",
    "made",
    "made_with",
    "has",
    "have",
    "include",
    "including",
    "ingredients",
}

TEXT_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
}


def is_ingredient_query(text):
    return any(w in INGREDIENT_TRIGGERS for w in text.split())


def extract_ingredient_segment(q):
    q = q.lower()
    patterns = [
        r"with (.+)",
        r"using (.+)",
        r"containing (.+)",
        r"made with (.+)",
        r"include (.+)",
        r"includes (.+)",
        r"has (.+)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            seg = m.group(1)
            seg = re.split(r"\b(in|under|about|for|ready)\b", seg)[0]
            return seg.strip()
    return ""


def extract_time_constraint(q):
    q = q.lower()
    for word, val in TEXT_NUMBERS.items():
        if f"{word} minute" in q or f"{word} min" in q:
            return val
    m = re.search(r"(\d+)\s*(minutes|min)", q)
    if m:
        return int(m.group(1))
    if "quick" in q:
        return 20
    return None


# ─────────────────────────────────────────────────────────────
# SEMANTIC INGREDIENT EXTRACTION (using ingredient vocabulary)
# ─────────────────────────────────────────────────────────────


def semantic_extract_ingredients(segment, top_k=10):
    q_emb = embed_text(segment)  # (768,)
    sims = ingredient_embs @ q_emb

    best = sims.max().item()
    threshold = max(0.30, best * 0.80)

    idxs = torch.topk(sims, top_k).indices.tolist()
    final = [all_ingredients[i] for i in idxs if sims[i].item() >= threshold]
    return final


# ─────────────────────────────────────────────────────────────
# FAST SEMANTIC INGREDIENT FILTERING (NO BERT)
# ─────────────────────────────────────────────────────────────


def filter_by_ingredients(df, emb_matrix, req_ingredients):
    if not req_ingredients:
        return df, emb_matrix

    # Encode required ingredients
    req_embs = torch.stack([embed_text(r) for r in req_ingredients])
    sims = ingredients_emb @ req_embs.T  # (N, K)
    min_sims = sims.min(dim=1).values  # (N)

    # ⚠ smart adaptive threshold
    best = min_sims.max().item()
    threshold = max(0.35, best * 0.65)

    mask = (min_sims >= threshold).numpy()

    if mask.sum() == 0:
        return df.iloc[0:0], emb_matrix[0:0]

    return df[mask], emb_matrix[mask]


def filter_by_time(df, emb_matrix, max_time):
    if max_time is None:
        return df, emb_matrix

    mask = (df["total_minutes"] <= max_time).to_numpy()

    # if nothing matches → return empty
    if mask.sum() == 0:
        return df.iloc[0:0], emb_matrix[0:0]

    return df[mask], emb_matrix[mask]


# ─────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────


def route_query(query):
    q = query.lower()

    # -------- 1) extract segment-based ingredients (with, using, etc.) --------
    seg = extract_ingredient_segment(q)
    semantic_ings = semantic_extract_ingredients(seg) if seg else []

    # -------- 2) fallback ingredient extraction using vocabulary --------
    # split by words
    tokens = re.findall(r"[a-zA-Z]+", q)

    # ingredient vocabulary in lowercase
    ing_vocab = set(i.lower() for i in all_ingredients)

    # ANY token that appears in your ingredient list is an ingredient
    fallback_ings = [t for t in tokens if t in ing_vocab]

    # -------- 3) merge, deduplicate, and keep only meaningful ones --------
    ingredients = list(set(semantic_ings + fallback_ings))

    return {
        "ingredients": ingredients,
        "time": extract_time_constraint(q),
        "semantic": query,
    }


# ─────────────────────────────────────────────────────────────
# SEARCH + RANKING
# ─────────────────────────────────────────────────────────────


def rank(q_emb, emb_matrix, top_k=5):
    sims = emb_matrix @ q_emb
    k = min(top_k, len(sims))
    idx = torch.topk(sims, k).indices.tolist()
    return idx


def search(query, top_k=5):
    info = route_query(query)

    print("Extracted prompt info:\n", info)

    # Use ingredient embedding if available
    if info["ingredients"]:
        q_text = " ".join(info["ingredients"])
    else:
        q_text = info["semantic"]

    q_emb = embed_text(q_text)

    # Filters (FAST)
    filtered_df, filtered_emb = filter_by_ingredients(df, full_emb, info["ingredients"])
    filtered_df, filtered_emb = filter_by_time(filtered_df, filtered_emb, info["time"])

    if len(filtered_df) == 0:
        return pd.DataFrame()

    idx = rank(q_emb, filtered_emb, top_k)
    return filtered_df.iloc[idx]


def format_recipe(row):
    return f"""
💡 {row.recipe_name}

🧂 Ingredients: {row.ingredients[:200]}...
👨‍🍳 Directions: {row.directions[:300]}...
⏱️ Time: {row.total_minutes} min
"""


if __name__ == "__main__":
    results = search("recipe with rice")
    print(results)

    for _, r in results.iterrows():
        print(format_recipe(r))
