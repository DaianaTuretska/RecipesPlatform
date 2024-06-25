# region				-----External Imports-----
import random
from django.core.management.base import BaseCommand
from catalog import models as catalog_models
from faker import Faker

# endregion


class Command(BaseCommand):
    help = "Generates records for each declared model"

    def add_arguments(self, parser):
        parser.add_argument("-n", "--number", nargs="?", type=int, default=50)

    def handle(self, *args, **options):
        fake = Faker()

        number = options.get("number")

        recipes_ids = catalog_models.Recipe.objects.values_list("id", flat=True)

        recipe_views = []

        for _ in range(number):
            recipe_view = catalog_models.RecipeView(
                recipe_id=random.choice(recipes_ids),
                ip_address=fake.ipv4(),
                created_at=fake.date_time_this_decade(),
            )
            recipe_views.append(recipe_view)

        catalog_models.RecipeView.objects.bulk_create(
            recipe_views, ignore_conflicts=True
        )
        self.stdout.write(self.style.SUCCESS("Data is successfully generated"))
