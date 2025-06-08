import os

from celery import Celery
from celery.schedules import crontab

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "website.settings")
app = Celery("website")

app.conf.beat_schedule = {
    "update-popular-recommendations": {
        "task": "update_popular_recommendations",
        # "schedule": crontab(hour=0, minute=0),
        "schedule": 10,
    },
    "update-sentiment-recommendations": {
        "task": "update_sentiment_recommendations",
        # "schedule": crontab(hour=0, minute=0),
        "schedule": 10,
    },
}


app.config_from_object("django.conf:settings", namespace="CELERY")

app.autodiscover_tasks()
