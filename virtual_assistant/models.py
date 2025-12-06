from django.utils import timezone
from django.db import models


class Message(models.Model):
    query = models.TextField()
    recipe_name = models.CharField(max_length=255)
    ingredients = models.JSONField()
    directions = models.TextField()
    total_time = models.FloatField()
    created_at = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(
        to="users.User",
        related_name="messages",
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    def __str__(self) -> str:
        return f"{self.query} - {self.author_id}"
