from django.urls import path
from .views import CreateCheckoutSessionView, ProductLandingPageView

from . import views

app_name = "catalog"

urlpatterns = [
    path("", views.home, name="recipe-home"),
    path(
        "recipe_details/<str:recipe_pk>",
        views.RecipeDetailsView.as_view(),
        name="recipe-details",
    ),
    path(
        "recipe_statistics/<str:recipe_pk>",
        views.RecipeStatisticsView.as_view(),
        name="recipe-statistics",
    ),
    path(
        "general_statistics/",
        views.GeneralStatisticsView.as_view(),
        name="general-statistics",
    ),
    path("search/", views.form_search, name="form-recipes"),
    path("recipes_search/", views.recipes_search, name="recipes-search"),
    path("reviews/<str:recipe_pk>/", views.ReviewView.as_view(), name="recipe-reviews"),
    path(
        "add_rating/<str:recipe_pk>/", views.RatingView.as_view(), name="recipe-rating"
    ),
    path(
        "add_to_saved_recipes/<str:recipe_pk>/",
        views.add_to_saved_recipes,
        name="add-to-saved-recipes",
    ),
    path(
        "delete_from_saved_recipes/<str:recipe_pk>/",
        views.delete_from_saved_recipes,
        name="delete-from-saved-recipes",
    ),
    path(
        "change_review_status/<str:review_id>/",
        views.ChangeReviewStatusView.as_view(),
        name="change-review-status",
    ),
    path("information/", views.InformationView.as_view(), name="information-page"),
    path("news/", views.news_page, name="news-page"),
    path("collections/", views.CollectionsPageView.as_view(), name="collections-page"),
    path("authors/", views.authors_page, name="authors-page"),
    path("donate/", ProductLandingPageView.as_view(), name="donate-page"),
    path(
        "create-checkout-session/",
        CreateCheckoutSessionView.as_view(),
        name="create-checkout-session",
    ),
    path("cancel/", views.DonateFailedView.as_view(), name="donate-failed"),
    path("success/", views.DonateSuccessView.as_view(), name="donate-success"),
]
