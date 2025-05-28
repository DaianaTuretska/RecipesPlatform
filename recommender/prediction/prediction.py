import os
import joblib
import numpy as np
from catalog.models import Recipe, UserSavedRecipe, RecipeView, Review, RecipeStatistic


def load_model_and_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "model.pkl")
    dataset_path = os.path.join(base_dir, "model", "dataset.pkl")

    model = joblib.load(model_path)
    dataset = joblib.load(dataset_path)
    return model, dataset


def get_seen_recipe_ids(user_id):
    seen_ids = set(
        RecipeStatistic.objects.filter(user_id=user_id).values_list(
            "recipe_id", flat=True
        )
    )
    seen_ids |= set(
        Review.objects.filter(user_id=user_id).values_list("recipe_id", flat=True)
    )
    seen_ids |= set(
        UserSavedRecipe.objects.filter(user_id=user_id).values_list(
            "recipe_id", flat=True
        )
    )
    return seen_ids


def recommend_recipes(user_id, top_n=5):
    model, dataset = load_model_and_dataset()
    user_mapping = dataset.mapping()[0]  # str(user_id) -> index
    item_mapping = dataset.mapping()[2]  # str(recipe_id) -> index
    item_id_reverse = {v: k for k, v in item_mapping.items()}  # index -> str(recipe_id)

    if str(user_id) not in user_mapping:
        print(f"User {user_id} not found in training data.")
        return Recipe.objects.none()

    user_index = user_mapping[str(user_id)]
    all_item_indices = list(item_mapping.values())

    # Filter out already seen items
    seen = get_seen_recipe_ids(user_id)
    unseen_item_indices = [
        idx for idx in all_item_indices if int(item_id_reverse[idx]) not in seen
    ]

    if not unseen_item_indices:
        return Recipe.objects.none()

    scores = model.predict(user_ids=user_index, item_ids=unseen_item_indices)
    top_indices = np.argsort(-scores)[:top_n]
    top_recipe_ids = [int(item_id_reverse[unseen_item_indices[i]]) for i in top_indices]

    return Recipe.objects.filter(id__in=top_recipe_ids)
