from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin,
)
from django.db import models

from utils.choices import Status
from .choices import Role


class CustomUserManager(BaseUserManager):
    def create_user(
        self, firstname, lastname, email, username, is_staff, password=None
    ):
        if not firstname:
            raise ValueError("Users must have a firstname")
        if not lastname:
            raise ValueError("Users must have a lastname")
        if not email:
            raise ValueError("Users must have an email")
        if not username:
            raise ValueError("Users must have an username")
        if not password:
            raise ValueError("Users must have a password")

        user = self.model(
            firstname=firstname,
            lastname=lastname,
            email=self.normalize_email(email),
            username=username,
            is_staff=is_staff,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, firstname, lastname, email, username, password=None):
        user = self.create_user(
            firstname=firstname,
            lastname=lastname,
            email=self.normalize_email(email),
            username=username,
            password=password,
        )
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.role = Role.ADMIN
        user.save(using=self._db)
        return user

    def update(self, user, firstname, lastname, email, username):
        user.firstname = firstname
        user.lastname = lastname
        user.email = email
        user.username = username
        user.save(using=self._db)


class User(AbstractBaseUser, PermissionsMixin):
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    email = models.EmailField(max_length=100, unique=True)
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)
    role = models.CharField(choices=Role.choices, default=Role.USER, max_length=10)
    status = models.CharField(
        choices=Status.choices, default=Status.ACTIVE, max_length=10
    )
    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    saved_recipes = models.ManyToManyField(
        to="catalog.Recipe",
        related_name="users_saved",
        through="catalog.UserSavedRecipe",
        blank=True,
        null=True,
    )

    objects = CustomUserManager()

    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = ["firstname", "lastname", "email", "password"]

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    def __str__(self):
        return self.username
