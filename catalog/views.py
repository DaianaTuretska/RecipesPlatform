import os
import json
import stripe
from collections import defaultdict
from django.contrib import messages
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
from django.db.models import (
    Q,
    Sum,
    Avg,
    F,
    Count,
    FilteredRelation,
    Subquery,
    OuterRef,
    Func,
    Value,
    FloatField,
    Subquery,
    CharField,
    Case,
    When,
)
from django.db.models.functions import Coalesce, TruncDay
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View
from django.views.generic import TemplateView
from django.core.serializers.json import DjangoJSONEncoder

from utils.cache import CacheStorage
from users.models import User
from .models import Recipe, RecipeStatistic, Review, Status, RecipeView, UserSavedRecipe
from . import forms
from . import service

CACHE_LIFETIME = int(os.getenv("CACHE_LIFETIME"))
cache_storage = CacheStorage(CACHE_LIFETIME)


stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


@login_required
def add_to_saved_recipes(request, recipe_pk):
    user = request.user
    recipe = Recipe.objects.filter(pk=recipe_pk, status=Status.ACTIVE).first()
    if not recipe:
        return redirect(home)

    user.saved_recipes.add(recipe)

    return redirect(reverse("catalog:recipe-details", args=(recipe_pk,)))


@login_required
def delete_from_saved_recipes(request, recipe_pk):
    user = request.user
    recipe = Recipe.objects.filter(pk=recipe_pk, status=Status.ACTIVE).first()
    if not recipe:
        return redirect(home)

    user.saved_recipes.remove(recipe)

    return redirect(reverse("catalog:recipe-details", args=(recipe_pk,)))


class RecipeDetailsView(View):
    def get(self, request, recipe_pk):
        if "recipe-details" in request.META["HTTP_REFERER"]:
            request.META["HTTP_REFERER"] = request.session.get("previous") or reverse(
                "recipe-home"
            )
        else:
            request.session["previous"] = request.META["HTTP_REFERER"]

        ip_address = request.META.get("REMOTE_ADDR")

        recipe = (
            Recipe.objects.filter(pk=recipe_pk, status=Status.ACTIVE)
            .annotate(rating=Avg("statistics__rating"))
            .annotate(total_saved=Count("users_saved", distinct=True))
            .first()
        )
        if not recipe:
            return redirect(home)

        RecipeView.objects.get_or_create(
            recipe=recipe,
            ip_address=ip_address,
            user_id=request.user.id,
        )

        return render(
            request,
            "recipe-details.html",
            {
                "recipe": recipe,
                "reviews": recipe.reviews.order_by("-created_at"),
                "form": forms.RecipeReviewForm(),
            },
        )


class RecipeStatisticsView(View):
    def get(self, request, recipe_pk):
        if "recipe-statistics" in request.META.get("HTTP_REFERER", {}):
            request.META["HTTP_REFERER"] = request.session.get("previous") or reverse(
                "recipe-home"
            )
        else:
            request.session["previous"] = request.META.get("HTTP_REFERER")

        recipe = (
            Recipe.objects.filter(pk=recipe_pk, status=Status.ACTIVE)
            .annotate(
                views_count=Count("views", distinct=True),
                saved_count=Count("user_saved_recipes", distinct=True),
                comments_count=Count("reviews", distinct=True),
                rating_count=Count("statistics", distinct=True),
            )
            .first()
        )

        if recipe.author != request.user:
            return redirect("catalog:recipe-home")

        ratings = []
        dates = []
        recipe_statistics_ratings = (
            RecipeStatistic.objects.filter(recipe=recipe)
            .annotate(date=TruncDay("created_at"))
            .values("date")
            .annotate(
                avg_rating=Avg(
                    "rating",
                    default=0,
                    distinct=True,
                    output_field=FloatField(),
                )
            )
            .values("date", "avg_rating")
        )

        for data in recipe_statistics_ratings:
            date = data.get("date").strftime("%Y-%m-%d")
            dates.append(date)

            ratings.append(data.get("avg_rating"))

        if not recipe:
            return redirect("catalog:recipe-home")

        context = {
            "recipe": recipe,
            "ratings": ratings,
            "dates": dates,
        }
        return render(request, "recipe-statistics.html", context=context)


class GeneralStatisticsView(View):
    def get(self, request):
        if "general-statistics" in request.META.get("HTTP_REFERER", {}):
            request.META["HTTP_REFERER"] = request.session.get("previous") or reverse(
                "recipe-home"
            )
        else:
            request.session["previous"] = request.META.get("HTTP_REFERER")

        recipes = (
            Recipe.objects.filter(status=Status.ACTIVE, author=request.user)
            .values("category")
            .annotate(
                avg_rating=Avg(
                    "statistics__rating",
                    distinct=True,
                    default=0,
                    output_field=FloatField(),
                )
            )
            .values("category", "avg_rating")
        )

        categories = []
        avg_ratings = []

        for recipe in recipes:
            categories.append(recipe.get("category"))
            avg_ratings.append(recipe.get("avg_rating"))

        saved_recipes = (
            UserSavedRecipe.objects.filter(
                recipe__in=Subquery(
                    Recipe.objects.filter(
                        status=Status.ACTIVE, author=request.user
                    ).values("id")
                )
            )
            .annotate(
                season=Case(
                    When(created_at__month__in=[12, 1, 2], then=Value("winter")),
                    When(created_at__month__in=[3, 4, 5], then=Value("spring")),
                    When(created_at__month__in=[6, 7, 8], then=Value("summer")),
                    When(created_at__month__in=[9, 10, 11], then=Value("autumn")),
                    output_field=CharField(),
                )
            )
            .values("season", "recipe__category")
            .annotate(count=Count("season"))
            .order_by("season", "recipe__category")
        )

        season_category_counts = defaultdict(
            lambda: {category: 0 for category in categories}
        )
        season_counts = {"winter": 0, "spring": 0, "summer": 0, "autumn": 0}

        for entry in saved_recipes:
            season = entry.get("season")
            category = entry.get("recipe__category")
            count = entry.get("count")
            season_category_counts[season][category] = count
            season_counts[season] += count

        seasons_categories = {}
        for season in season_counts:
            seasons_categories[season] = season_category_counts.get(season, {})

        context = {
            "categories": categories,
            "avg_ratings": avg_ratings,
            "season_category_counts": seasons_categories,
            "seasons_counts": list(season_counts.values()),
        }

        return render(request, "general-statistics.html", context=context)


class ReviewView(View):
    queryset = Review.objects

    def get_reviews(self, recipe_pk):
        return (
            self.queryset.filter(recipe=recipe_pk)
            .annotate(
                user_firstname=F("user__firstname"),
                user_lastname=F("user__lastname"),
            )
            .order_by("-created_at")
            .values()
        )

    def get(self, request, recipe_pk):
        if not request.user.is_authenticated:
            return JsonResponse({"message": "Not authorized"})
        return JsonResponse(
            {
                "reviews": json.dumps(
                    list(self.get_reviews(recipe_pk)), cls=DjangoJSONEncoder
                )
            }
        )

    def post(self, request, recipe_pk):
        error = ""
        if not request.user.is_authenticated:
            return JsonResponse({"message": "Not authorized"})
        text = request.POST.get("text-comment")
        if text.strip():
            recipe = Recipe.objects.filter(pk=recipe_pk, status=Status.ACTIVE).first()
            if not recipe:
                return redirect(home)
            Review.objects.create(
                user=request.user,
                recipe=recipe,
                comment=text,
            )
        else:
            error = "The comment field should not be blank."

        return JsonResponse(
            {
                "reviews": json.dumps(
                    list(self.get_reviews(recipe_pk)), cls=DjangoJSONEncoder
                ),
                "error": error,
            }
        )


class RatingView(View):
    def post(self, request, recipe_pk):
        data = json.loads(request.body)
        recipe = Recipe.objects.filter(pk=recipe_pk, status=Status.ACTIVE).first()
        if not recipe:
            return redirect(home)
        RecipeStatistic.objects.update_or_create(
            recipe=recipe,
            user=request.user,
            defaults={"rating": data.get("rating", 3)},
        )

        return HttpResponseRedirect(
            redirect_to=reverse("catalog:recipe-details", args=[recipe_pk])
        )


class ChangeReviewStatusView(View):
    def post(self, request, review_id):
        review = Review.objects.filter(pk=review_id).first()
        if review:
            review.status = request.POST.get("status")
            review.save()

        return HttpResponse("Success", content_type="text/plain")


def home(request):
    if request.user.is_authenticated:
        top_recipes_ids = cache.get(f"user:{request.user.id}:recs:popular", [])
        recipes_to_like_ids = cache.get(f"user:{request.user.id}:recs:sentiment", [])

        top_recipes = Recipe.objects.filter(id__in=top_recipes_ids)
        recipes_to_like = Recipe.objects.filter(id__in=recipes_to_like_ids)
    else:
        top_recipes = cache_storage.get_value("top_recipes")
        recipes_to_like = []

    new_recipes = cache_storage.get_value("new_recipes")
    users_recipes = cache_storage.get_value("users_recipes")
    cuisines = cache_storage.get_value("cuisines")

    if top_recipes is None:
        top_recipes = (
            Recipe.objects.filter(status=Status.ACTIVE)
            .annotate(rating=Avg("statistics__rating"))
            .order_by("-rating")[:10]
        )
        cache_storage.add_value("top_recipes", top_recipes)

    if new_recipes is None:
        new_recipes = Recipe.objects.filter(status=Status.ACTIVE).order_by("-pk")[:10]
        cache_storage.add_value("new_recipes", new_recipes)

    if users_recipes is None:
        users_recipes = Recipe.objects.filter(
            status=Status.ACTIVE, author__isnull=False
        ).order_by("-pk")[:10]
        cache_storage.add_value("users_recipes", users_recipes)

    if cuisines is None:
        cuisines = Recipe.objects.values_list("cuisine", flat=True)
        cache_storage.add_value("cuisines", cuisines)

    return render(
        request,
        "home.html",
        {
            "top_recipes": top_recipes,
            "recipes_to_like": recipes_to_like,
            "new_recipes": new_recipes,
            "users_recipes": users_recipes,
            "cuisines": cuisines,
        },
    )


class InformationView(TemplateView):
    template_name = "information_page.html"


def recipes_search(request):
    cuisine = request.GET.get("cuisine")
    category = request.GET.get("category")
    recipes = Recipe.objects.filter(
        Q(cuisine=cuisine) | Q(category=category), status=Status.ACTIVE
    )
    return render(
        request, "search.html", {"recipes": recipes, "q": cuisine or category}
    )


def form_search(request):
    q = request.GET.get("searchbar")
    recipes = []

    if q and q.strip() == "Vegetarian":
        titles = [
            "Spring Salad (Zelenyj Salat)",
            "Vinegret Salad",
            "Syrnyk (Sweet Cheese Pancakes)",
            "Pampushky (Garlic Bread Rolls)",
            "Honey Cake (Medovik)",
            "Sorrel Soup (Green Borsch)",
            "Apple Cake (Yabluchnyk)",
            "Caprese Salad",
            "Mango Sticky Rice",
            "Deruny",
            "Varenuku",
        ]
        recipes = Recipe.objects.filter(title__in=titles).order_by("title")
    elif q:
        search_terms = q.split()
        q_objects = []

        for term in search_terms:
            q_objects.append(
                Q(author__firstname__icontains=term)
                | Q(author__lastname__icontains=term)
                | Q(title__icontains=term)
                | Q(year__icontains=term)
            )

        recipes = Recipe.objects.filter(*q_objects)

    return render(request, "search.html", {"recipes": recipes, "q": q})


def news_page(request):
    news = service.news_parse()
    return render(request, "news_page.html", {"news": news})


class CollectionsPageView(View):
    def get(self, request):
        recipes_total_saved = cache_storage.get_value("recipes_total_saved")
        most_discsused_recipes = cache_storage.get_value("most_discsused_recipes")

        if recipes_total_saved is None:
            recipes_total_saved = (
                Recipe.objects.filter(status=Status.ACTIVE)
                .annotate(total_saved=Count("users_saved"))
                .order_by("-total_saved")
            )[:10]
            cache_storage.add_value("recipes_total_saved", recipes_total_saved)

        if most_discsused_recipes is None:
            most_discsused_recipes = (
                Recipe.objects.filter(status=Status.ACTIVE)
                .annotate(
                    active_reviews=FilteredRelation(
                        "reviews", condition=Q(reviews__status=Status.ACTIVE)
                    )
                )
                .annotate(total_reviews=Count("active_reviews"))
                .order_by("-total_reviews")
            )[:10]

            cache_storage.add_value("most_discsused_recipes", most_discsused_recipes)

        return render(
            request,
            "collections.html",
            {
                "recipes_total_saved": recipes_total_saved,
                "most_discsused_recipes": most_discsused_recipes,
            },
        )


def authors_page(request):
    # authors = cache_storage.get_value("authors")
    authors = None

    if authors is None:
        authors = (
            (
                User.objects.filter(status=Status.ACTIVE)
                .annotate(recipes_count=Count("recipes"))
                .filter(recipes_count__gte=1)
                .annotate(
                    total_rating=Sum("recipes__statistics__rating"),
                    average_rating=Avg("recipes__statistics__rating"),
                    total_saved=Subquery(
                        Recipe.objects.filter(author=OuterRef("pk"))
                        .annotate(
                            total_saved=Coalesce(
                                Func("users_saved", function="Count", distinct=True),
                                Value(0),
                            )
                        )
                        .values("total_saved")
                    ),
                )
            )
            .filter(total_rating__gt=0)
            .order_by("-average_rating")
        )
        cache_storage.add_value("authors", authors)
    return render(request, "authors.html", {"authors": authors})


class DonateSuccessView(TemplateView):
    template_name = "success.html"


class DonateFailedView(TemplateView):
    template_name = "cancel.html"


class ProductLandingPageView(TemplateView):
    template_name = "donate_page.html"

    def get_context_data(self, **kwargs):
        context = super(ProductLandingPageView, self).get_context_data(**kwargs)
        context.update({"STRIPE_PUBLIC_KEY": os.getenv("STRIPE_PUBLIC_KEY")})
        return context


class CreateCheckoutSessionView(View):
    def post(self, request, *args, **kwargs):
        data = json.loads(request.body)
        unit_amount = data.get("unit_amount")
        if str(unit_amount).isdigit():
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[
                    {
                        "price_data": {
                            "currency": "usd",
                            "unit_amount": unit_amount,
                            "product_data": {
                                "name": "DONATE",
                            },
                        },
                        "quantity": 1,
                    },
                ],
                mode="payment",
                success_url=request.build_absolute_uri(
                    reverse("catalog:donate-success")
                ),
                cancel_url=request.build_absolute_uri(reverse("catalog:donate-failed")),
            )
            return JsonResponse({"id": checkout_session.id})
        else:
            messages.error(request, "Wrong unit amount.")
            return JsonResponse({})
