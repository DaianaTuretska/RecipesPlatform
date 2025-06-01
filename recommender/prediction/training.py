import os
import joblib
from lightfm import LightFM
from lightfm.data import Dataset
from catalog.models import RecipeStatistic, Review, UserSavedRecipe, RecipeView

import math
from django.utils import timezone


def collect_interactions_binary_and_weights():
    """
    Returns two parallel lists:
      - binary_rows:  [(user_id_str, recipe_id_str, 1.0), ...]
      - weight_rows:  [(user_id_str, recipe_id_str, confidence_float), ...]
    """
    binary_rows = []
    weight_rows = []

    # 1) RecipeStatistic → use normalized rating as confidence, binary = 1
    for stat in RecipeStatistic.objects.select_related("user", "recipe"):
        uid = str(stat.user_id)
        rid = str(stat.recipe_id)

        # Normalize rating into [0..1]; if your ratings are out of 5, divide by 5.0
        normalized_rating = float(stat.rating) / 5.0

        binary_rows.append((uid, rid, 1.0))
        weight_rows.append((uid, rid, normalized_rating))

    # 2) Review → base confidence 0.6 (example); binary = 1
    BASE_REVIEW = 0.6
    for review in Review.objects.select_related("user", "recipe"):
        uid = str(review.user_id)
        rid = str(review.recipe_id)

        # Optionally apply a time decay on reviews:
        #   age_days = (now − created_at).days
        #   decay = math.exp(-DECAY_RATE * age_days)
        #   confidence = BASE_REVIEW * decay
        age_days = (timezone.now() - review.created_at).days
        DECAY_RATE = 0.005  # tweak as you like
        confidence = BASE_REVIEW * math.exp(-DECAY_RATE * age_days)

        binary_rows.append((uid, rid, 1.0))
        weight_rows.append((uid, rid, confidence))

    # 3) Saved recipes → base confidence 0.4; binary = 1
    BASE_SAVE = 0.4
    for saved in UserSavedRecipe.objects.select_related("user", "recipe"):
        uid = str(saved.user_id)
        rid = str(saved.recipe_id)

        age_days = (timezone.now() - saved.created_at).days
        DECAY_RATE = 0.005
        confidence = BASE_SAVE * math.exp(-DECAY_RATE * age_days)

        binary_rows.append((uid, rid, 1.0))
        weight_rows.append((uid, rid, confidence))

    # 4) Recipe views → base confidence 0.2; binary = 1
    BASE_VIEW = 0.2
    for view in RecipeView.objects.filter(user__isnull=False).select_related(
        "user", "recipe"
    ):
        uid = str(view.user_id)
        rid = str(view.recipe_id)

        age_days = (timezone.now() - view.created_at).days
        DECAY_RATE = 0.005
        confidence = BASE_VIEW * math.exp(-DECAY_RATE * age_days)

        binary_rows.append((uid, rid, 1.0))
        weight_rows.append((uid, rid, confidence))

    return binary_rows, weight_rows


def train_lightfm_model(output_dir=None) -> None:
    # Save to a folder inside the current script's directory
    if output_dir is None:
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # current file's directory
        output_dir = os.path.join(base_dir, "model")

    binary_interactions, weight_interactions = collect_interactions_binary_and_weights()

    if not binary_interactions:
        print("⚠️ No interactions found. Training aborted.")
        return

    # 2) Extract unique user IDs and recipe IDs from the binary rows
    user_ids = {u for u, _, _ in binary_interactions}
    recipe_ids = {i for _, i, _ in binary_interactions}

    # 3) Fit the Dataset object
    dataset = Dataset()
    dataset.fit(users=user_ids, items=recipe_ids)

    # 4) Build the two sparse matrices:
    #    - R_sparse:  binary (0/1) interactions
    #    - W_sparse:  confidence‐weight interactions (same shape as R_sparse)
    R_sparse, _ = dataset.build_interactions(binary_interactions)
    W_sparse, _ = dataset.build_interactions(weight_interactions)

    # 5) Instantiate and train LightFM with sample_weight=…
    model = LightFM(loss="logistic")  # you can also try "bpr" or "logistic"
    model.fit(
        interactions=R_sparse,
        sample_weight=W_sparse,
        epochs=30,  # you can bump this up to 100 if you have time/budget
        num_threads=4,  # parallelism
    )

    # 6) Save both model + dataset so you can reuse them in your recommender
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    joblib.dump(dataset, os.path.join(output_dir, "dataset.pkl"))

    print(f"✅ LightFM model trained and saved to: {output_dir}")
