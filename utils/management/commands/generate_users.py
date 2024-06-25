# region				-----External Imports-----
from django.core.management.base import BaseCommand
from users import models as user_models
from users import choices as user_choices
from utils import choices as utils_choices
from faker import Faker
from django.contrib.auth.hashers import make_password
from django.contrib.auth import models as django_auth_models

# endregion


class Command(BaseCommand):
    help = "Generates records for each declared model"

    def add_arguments(self, parser):
        parser.add_argument("-n", "--number", nargs="?", type=int, default=50)

    def handle(self, *args, **options):
        fake = Faker()

        number = options.get("number")
        group = django_auth_models.Group.objects.filter(
            name="Base User Add Recipes"
        ).first()  # ? you should create group with exact name

        for _ in range(number):
            profile = fake.profile()
            user = user_models.User(
                firstname=fake.first_name(),
                lastname=fake.last_name(),
                email=profile.get("mail"),
                username=profile.get("username"),
                password=make_password("password"),
                role=user_choices.Role.USER,
                status=utils_choices.Status.ACTIVE,
                is_staff=True,
            )
            user.save()
            user.groups.add(group)

        self.stdout.write(self.style.SUCCESS("Data is successfully generated"))
