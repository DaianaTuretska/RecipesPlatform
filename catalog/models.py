from django.utils import timezone
from django.db import models
from ckeditor.fields import RichTextField

from utils.choices import Status


class Recipe(models.Model):
    YEAR_CHOICES = [(y, y) for y in range(1984, timezone.now().today().year + 1)]

    title = models.CharField(max_length=255)
    description = RichTextField(blank=True, null=True)
    ingredients = RichTextField(blank=True, null=True)
    cooking_method = RichTextField(blank=True, null=True)
    cuisine = models.CharField(max_length=255)
    category = models.CharField(max_length=255)
    year = models.IntegerField(choices=YEAR_CHOICES, default=timezone.now().year)
    status = models.CharField(
        choices=Status.choices, default=Status.ACTIVE, max_length=100
    )
    image = models.ImageField(upload_to="media/recipe images/%Y-%m-%d")
    author = models.ForeignKey(
        to="users.User",
        related_name="recipes",
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )

    def __str__(self) -> str:
        return self.title


class VideoLink(models.Model):
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=255)
    recipe = models.ForeignKey(
        to="catalog.Recipe",
        related_name="video_links",
        on_delete=models.CASCADE,
    )


class RecipeStatistic(models.Model):
    recipe = models.ForeignKey(
        to="catalog.Recipe",
        related_name="statistics",
        on_delete=models.CASCADE,
    )

    user = models.ForeignKey(
        to="users.User",
        related_name="statistics",
        on_delete=models.CASCADE,
    )
    rating = models.DecimalField(
        default=2.5,
        max_digits=3,
        decimal_places=2,
    )
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ("recipe", "user")


class Review(models.Model):
    user = models.ForeignKey(
        to="users.User",
        related_name="reviews",
        on_delete=models.CASCADE,
    )

    recipe = models.ForeignKey(
        to="catalog.Recipe",
        related_name="reviews",
        on_delete=models.CASCADE,
    )

    status = models.CharField(
        choices=Status.choices,
        default=Status.ACTIVE,
        max_length=10,
    )
    comment = RichTextField()
    created_at = models.DateTimeField(auto_now_add=True)


class UserSavedRecipe(models.Model):
    user = models.ForeignKey(
        to="users.User",
        related_name="user_saved_recipes",
        on_delete=models.CASCADE,
    )

    recipe = models.ForeignKey(
        to="catalog.Recipe",
        related_name="user_saved_recipes",
        on_delete=models.CASCADE,
    )

    created_at = models.DateTimeField(default=timezone.now)


class RecipeView(models.Model):
    recipe = models.ForeignKey(
        "catalog.Recipe", related_name="views", on_delete=models.CASCADE
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = ("recipe", "ip_address")

    def __str__(self):
        return f"{self.id}"
