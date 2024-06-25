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

        user_saved_recipes = []

        for _ in range(number):
            user_saved_recipe = catalog_models.UserSavedRecipe(
                recipe_id=random.choice(recipes_ids),
                user_id=random.choice(users_ids),
                created_at=fake.date_time_this_decade(),
            )
            user_saved_recipes.append(user_saved_recipe)

        catalog_models.UserSavedRecipe.objects.bulk_create(
            user_saved_recipes, ignore_conflicts=True
        )
        self.stdout.write(self.style.SUCCESS("Data is successfully generated"))
