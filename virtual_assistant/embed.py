import torch.nn.functional as F
import torch
import os
from dataset import prepare_recipe_dataset
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
MODEL_PATH = os.path.join(CURRENT_DIR, "distilbert_recipe_finetuned")
ASSETS_PATH = os.path.join(CURRENT_DIR, "assets")
MULTI_EMBEDDINGS_PATH = os.path.join(ASSETS_PATH, "recipe_embeddings_multi.pt")
DATA_FRAME_PATH = os.path.join(ASSETS_PATH, "recipe_df.pkl")
ALL_INGREDIENTS = os.path.join(ASSETS_PATH, "all_ingredients_list.pt")
ALL_INGREDIENTS_EMBEDDINGS = os.path.join(ASSETS_PATH, "ingredient_embs.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

BATCH_SIZE = 128


def embed_batch(texts):
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)

    hidden = outputs.last_hidden_state
    attn = tokens["attention_mask"].unsqueeze(-1)

    masked = hidden * attn
    summed = masked.sum(dim=1)
    counts = attn.sum(dim=1).clamp(min=1)
    pooled = summed / counts

    pooled = F.normalize(pooled, dim=1)
    return pooled.cpu()


def embed_text(text: str):
    return embed_batch([text])[0]


def embed_column(texts):
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        all_embs.append(embed_batch(batch))
    return torch.cat(all_embs, dim=0)


def embed_fields(df):
    names = embed_column(df["recipe_name"].tolist())

    # better ingredient texts
    ingredients_texts = [" ".join(lst) for lst in df["ingredient_list"]]
    ingredients = embed_column(ingredients_texts)

    directions = embed_column(df["directions"].tolist())
    full = embed_column(df["document_text"].tolist())

    return {
        "name_emb": names,
        "ingredients_emb": ingredients,
        "directions_emb": directions,
        "full_emb": full,
    }


def embed_ingredients(all_ingredients):
    batch_size = 64
    all_embs = []

    for i in range(0, len(all_ingredients), batch_size):
        batch = all_ingredients[i : i + batch_size]
        emb = embed_batch(batch)  # (B, 768)
        all_embs.append(emb)

    ingredient_embs = torch.cat(all_embs, dim=0)  # (N, 768)
    return ingredient_embs


if __name__ == "__main__":
    df = prepare_recipe_dataset()
    print("DATASET READY")

    # embeddings = embed_fields(df)
    # torch.save(embeddings, MULTI_EMBEDDINGS_PATH)

    # df.reset_index(drop=True, inplace=True)
    # df.to_pickle(DATA_FRAME_PATH)

    unique_ingredients = sorted(
        {ing.lower() for lst in df["ingredient_list"] for ing in lst}
    )
    all_ing_list = list(unique_ingredients)
    torch.save(all_ing_list, os.path.join(CURRENT_DIR, ALL_INGREDIENTS))
    ingredient_embs = embed_ingredients(all_ing_list)
    torch.save(ingredient_embs, ALL_INGREDIENTS_EMBEDDINGS)

    print("EMBEDDINGS READY")
