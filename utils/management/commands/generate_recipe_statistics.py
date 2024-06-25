# region				-----External Imports-----
import random
from django.core.management.base import BaseCommand
from users import models as users_models
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
        users_ids = users_models.User.objects.values_list("id", flat=True)

        recipe_statistics = []

        for _ in range(number):
            recipe_statistic = catalog_models.RecipeStatistic(
                recipe_id=random.choice(recipes_ids),
                user_id=random.choice(users_ids),
                rating=fake.pyint(min_value=1, max_value=5),
                created_at=fake.date_time_this_decade(),
            )
            recipe_statistics.append(recipe_statistic)

        catalog_models.RecipeStatistic.objects.bulk_create(
            recipe_statistics, ignore_conflicts=True
        )
        self.stdout.write(self.style.SUCCESS("Data is successfully generated"))
