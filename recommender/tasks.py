from celery import shared_task
from django.core.cache import cache
from django.db.models import Count, Avg

from .prediction.training import train_lightfm_model
from .prediction.prediction import recommend_recipes  # popularity / CF
from .llm.distilbert.recommend import (
    load_sentiment_pipeline,
    recommend_recipes_for_user,
)
from users.models import User
from catalog.models import Recipe, Review
from utils.choices import Status


TOP_N = 10
CACHE_TTL = 24 * 3600  # ? 1 day

POPULAR_KEY = "user:{user_id}:recs:popular"
SENTIMENT_KEY = "user:{user_id}:recs:sentiment"


@shared_task(name="update_popular_recommendations")
def update_popular_recommendations():
    """
    Recompute & cache the 'most popular / CF‐based' recs for each user.
    """
    # (re)train your collaborative/popular model
    train_lightfm_model()

    for user in User.objects.filter(is_active=True):
        try:
            qs = recommend_recipes(user.id, top_n=TOP_N)
            ids = list(qs.values_list("id", flat=True))
            cache.set(POPULAR_KEY.format(user_id=user.id), ids, CACHE_TTL)
        except Exception as e:
            print(f"[popular] user {user.id} failed: {e}")

    print("✅ Popular Recommendations Updated")


@shared_task(name="update_sentiment_recommendations")
def update_sentiment_recommendations():
    """
    Recompute & cache the 'most positively commented' recs for each user.
    """
    pipe = load_sentiment_pipeline()
    for user in User.objects.filter(is_active=True):
        try:
            qs = recommend_recipes_for_user(pipe, user.id, top_n=TOP_N)
            ids = [recipe.id for recipe in qs]
            cache.set(SENTIMENT_KEY.format(user_id=user.id), ids, CACHE_TTL)
        except Exception as e:
            print(f"[sentiment] user {user.id} failed: {e}")

    print("✅ Sentiment Recommendations Updated")
