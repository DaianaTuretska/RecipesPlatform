from typing import Any
from django.contrib import admin
from django.db.models.query import QuerySet
from django.http import HttpRequest

from . import models


@admin.register(models.Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "query",
        "recipe_name",
        "ingredients",
        "directions",
        "author",
        "created_at",
    ]
    list_filter = ["author", "created_at"]
    list_select_related = ["author"]
