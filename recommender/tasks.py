from celery import shared_task
from .prediction.training import train_lightfm_model
from .prediction.prediction import recommend_recipes
from users.models import User
from django.core.cache import cache

TOP_N = 5  # number of recommendations per user
CACHE_TTL = 24 * 3600  # 24 hours in seconds


@shared_task(name="update_recommendations")
def update_recommendations() -> None:
    train_lightfm_model()

    for user in User.objects.all():
        try:
            recommendations = recommend_recipes(user.id, top_n=TOP_N)
            cache.set(
                f"user:{user.id}:recommendations",
                list(recommendations.values_list("id", flat=True)),
                CACHE_TTL,
            )

        except Exception as e:
            print(f"⚠️ Failed to cache recommendations for user {user.id}: {e}")

    print("Updated recommendations cache")
