# region				-----External Imports-----
import random
from django.core.management.base import BaseCommand
from users import models as users_models
from catalog import models as catalog_models
from utils import choices as utils_choices
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

        recipe_reviews = []

        for _ in range(number):
            recipe_review = catalog_models.Review(
                recipe_id=random.choice(recipes_ids),
                user_id=random.choice(users_ids),
                status=utils_choices.Status.ACTIVE,
                comment=fake.text(max_nb_chars=200),
            )
            recipe_reviews.append(recipe_review)

        catalog_models.Review.objects.bulk_create(recipe_reviews, ignore_conflicts=True)
        self.stdout.write(self.style.SUCCESS("Data is successfully generated"))
