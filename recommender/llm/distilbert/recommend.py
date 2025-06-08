import os
import glob
import torch
from collections import Counter
from itertools import chain
from django.db.models import Avg
from django.core.management.base import BaseCommand

import pandas as pd
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TextClassificationPipeline,
)
from safetensors.torch import load_file as load_safetensors

from users.models import User
from catalog.models import Recipe, Review, RecipeStatistic
from utils.choices import Status


ROOT_DIR = os.path.join(os.path.dirname(__file__), "distilbert_three_class")


# 1) Utility to load your latest checkpoint + build a HF pipeline
def load_sentiment_pipeline(max_length: int = 128) -> TextClassificationPipeline:
    # locate latest checkpoint
    ckpts = glob.glob(os.path.join(ROOT_DIR, "checkpoint-*"))
    latest = max(ckpts, key=lambda d: int(d.rsplit("-", 1)[-1]))
    print(f"[sentiment] loading from {latest}")

    # load model weights
    model = DistilBertForSequenceClassification.from_pretrained(
        latest, from_tf=False, use_safetensors=True
    )
    state = load_safetensors(os.path.join(latest, "model.safetensors"), device="cpu")
    model.load_state_dict(state)
    model.eval()

    # tokenizer + pipeline
    tokenizer = DistilBertTokenizerFast.from_pretrained(latest)
    device = 0 if torch.cuda.is_available() else -1

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device,
        truncation=True,
        max_length=max_length,
        return_all_scores=False,  # only top label
    )
    return pipe


# 2) Recommendation logic
def recommend_recipes_for_user(pipe, user_id: int, top_n: int = 5):
    user = User.objects.get(pk=user_id)
    reviews = Review.objects.filter(user=user, status=Status.ACTIVE).select_related(
        "recipe"
    )

    comments = [r.comment for r in reviews]
    preds = pipe(comments, batch_size=32)

    positive_recipes = [
        rev.recipe for rev, pred in zip(reviews, preds) if pred["label"] == "LABEL_2"
    ]

    # pick top categories
    if positive_recipes:
        from collections import Counter

        top_cats = [
            cat
            for cat, _ in Counter(r.category for r in positive_recipes).most_common(2)
        ]
        seen_ids = [r.recipe.id for r in reviews]

        # 1) get your category‐based slice
        cat_qs = list(
            Recipe.objects.filter(category__in=top_cats, status=Status.ACTIVE)
            .exclude(id__in=seen_ids)
            .annotate(avg_rating=Avg("statistics__rating"))
            .order_by("-avg_rating")[:top_n]
        )
    else:
        cat_qs = []

    # 2) if that’s fewer than you asked for, pull in from your global fallback
    if len(cat_qs) < top_n:
        exclude_ids = seen_ids + [r.id for r in cat_qs]
        fallback_qs = list(
            Recipe.objects.annotate(avg_rating=Avg("statistics__rating"))
            .exclude(id__in=exclude_ids)
            .order_by("-avg_rating")[: (top_n - len(cat_qs))]
        )
    else:
        fallback_qs = []

    # 3) stitch them together and return exactly top_n
    combined = list(chain(cat_qs, fallback_qs))
    return combined[:top_n]
