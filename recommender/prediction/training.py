import os
import joblib
from lightfm import LightFM
from lightfm.data import Dataset
from catalog.models import RecipeStatistic, Review, UserSavedRecipe, RecipeView


def collect_interactions() -> list[tuple[str, str, float]]:
    interactions = []

    for stat in RecipeStatistic.objects.select_related("user", "recipe"):
        interactions.append(
            (str(stat.user_id), str(stat.recipe_id), float(stat.rating))
        )

    for review in Review.objects.select_related("user", "recipe"):
        interactions.append((str(review.user_id), str(review.recipe_id), 2.0))

    for saved in UserSavedRecipe.objects.select_related("user", "recipe"):
        interactions.append((str(saved.user_id), str(saved.recipe_id), 1.5))

    for view in RecipeView.objects.filter(user__isnull=False).select_related(
        "user", "recipe"
    ):
        interactions.append((str(view.user_id), str(view.recipe_id), 1.0))

    return interactions


def train_lightfm_model(output_dir=None) -> None:
    # Save to a folder inside the current script's directory
    if output_dir is None:
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # current file's directory
        output_dir = os.path.join(base_dir, "model")

    interactions = collect_interactions()

    if not interactions:
        print("⚠️ No interactions found. Training aborted.")
        return None

    user_ids = {u for u, _, _ in interactions}
    recipe_ids = {r for _, r, _ in interactions}

    dataset = Dataset()
    dataset.fit(users=user_ids, items=recipe_ids)

    interactions_matrix, _ = dataset.build_interactions(interactions)
    model = LightFM(loss="logistic")
    model.fit(interactions_matrix, epochs=100, num_threads=1)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    joblib.dump(dataset, os.path.join(output_dir, "dataset.pkl"))

    print(f"✅ LightFM model trained and saved to: {output_dir}")
