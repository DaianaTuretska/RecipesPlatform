from typing import Any
from django.contrib import admin
from django.db.models.query import QuerySet
from django.http import HttpRequest
from users import choices as users_choices

from . import models


class InlineUserSavedRecipeAdmin(admin.TabularInline):
    model = models.UserSavedRecipe
    extra = 0


class InlineVideoLinkAdmin(admin.TabularInline):
    model = models.VideoLink
    extra = 0


@admin.register(models.Recipe)
class RecipeAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "cuisine", "category", "year", "status", "author"]
    list_filter = ["cuisine", "category", "year", "status", "author"]
    inlines = [InlineVideoLinkAdmin, InlineUserSavedRecipeAdmin]

    def get_queryset(self, request: HttpRequest) -> QuerySet[Any]:
        queryset = super().get_queryset(request)

        if (
            not request.user.is_superuser
            or not request.user.role == users_choices.Role.ADMIN
        ):
            return queryset.filter(author=request.user)

        return queryset

    def has_add_permission(self, request):
        return super().has_add_permission(request)

    def has_delete_permission(self, request, obj=None):
        if (
            not request.user.is_superuser
            or not request.user.role == users_choices.Role.ADMIN
        ):
            if obj and obj.author != request.user:
                return False

        return super().has_delete_permission(request)

    def has_change_permission(self, request, obj=None):
        if (
            not request.user.is_superuser
            or not request.user.role == users_choices.Role.ADMIN
        ):
            if obj and obj.author != request.user:
                return False

        return super().has_change_permission(request)


@admin.register(models.Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ["id", "recipe", "user", "created_at"]
    list_filter = ["recipe", "user", "created_at"]
    list_select_related = ["recipe", "user"]


@admin.register(models.RecipeStatistic)
class RecipeStatisticAdmin(admin.ModelAdmin):
    list_display = ["id", "recipe", "user", "rating"]


@admin.register(models.RecipeView)
class RecipeViewAdmin(admin.ModelAdmin):
    list_display = ["id", "recipe", "user", "ip_address", "created_at"]
    list_select_related = ["recipe", "user"]


@admin.register(models.UserSavedRecipe)
class UserSavedRecipeAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "recipe", "created_at"]
    list_select_related = ["recipe", "user"]
