import os
import json
import string
import random
import typing
from django.views import View
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.generic import TemplateView
from django.contrib.auth import authenticate, login, logout, models as auth_models
from django.core.mail import send_mail
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.db.models import Q

from catalog.models import Recipe
from utils.choices import Status
from .forms import ChangePasswordForm, EditProfileForm, RegistrationForm, ContactForm
from .models import User


class ProfileDetailsView(TemplateView):
    template_name = "profile_details.html"


class ProfileEditView(View):
    def get(self, request):
        user = request.user
        data = {
            "user_id": user.pk,
            "firstname": user.firstname,
            "lastname": user.lastname,
            "email": user.email,
            "username": user.username,
        }
        form = EditProfileForm(initial=data)
        return render(self.request, "profile_edit.html", {"form": form})

    def post(self, request):
        form = EditProfileForm(self.request.POST)
        if form.is_valid():
            firstname = form.cleaned_data.get("firstname")
            lastname = form.cleaned_data.get("lastname")
            email = form.cleaned_data.get("email")
            username = form.cleaned_data.get("username")
            User.objects.update(
                user=request.user,
                firstname=firstname,
                lastname=lastname,
                email=email,
                username=username,
            )
            return redirect(reverse("users:profile-details"))
        return render(self.request, "profile_edit.html", {"form": form})


@login_required
def profile_saved_recipes(request):
    saved_recipes = request.user.saved_recipes.all()
    saved_recipes_values = saved_recipes.values("pk", "cuisine")
    pks = saved_recipes_values.values_list("pk", flat=True)
    cuisines = saved_recipes_values.values_list("cuisine", flat=True).distinct()
    recommended_recipes = (
        Recipe.objects.filter(cuisine__in=cuisines, status=Status.ACTIVE)
        .exclude(pk__in=pks)
        .order_by("?")[:10]
    )
    return render(
        request,
        "profile_saved_recipes.html",
        {"recommended_recipes": recommended_recipes, "saved_recipes": saved_recipes},
    )


@login_required
def profile_my_recipes(request):
    my_recipes = Recipe.objects.filter(author=request.user, status=Status.ACTIVE)
    return render(
        request,
        "profile_my_recipes.html",
        {"my_recipes": my_recipes},
    )


@login_required
def change_password(request):
    user = request.user
    if request.method == "POST":
        form = ChangePasswordForm(request.POST)
        if form.is_valid():
            old_password = form.cleaned_data.get("old_password")
            if user and user.check_password(old_password):
                new_password = form.cleaned_data.get("new_password")
                user.set_password(new_password)
                user.save()
                messages.success(request, "Password successfully updated.")
                return redirect(reverse("users:profile-details"))
            else:
                messages.error(request, "Wrong old password.")
    else:
        form = ChangePasswordForm()
    return render(request, "change_password.html", {"form": form})


def registration(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")

            registered_user = User.objects.create_user(
                firstname=form.cleaned_data.get("firstname"),
                lastname=form.cleaned_data.get("lastname"),
                username=username,
                email=form.cleaned_data.get("email"),
                password=password,
                is_staff=True,
            )

            base_user_add_recipe_group = auth_models.Group.objects.filter(id=1).first()
            registered_user.groups.add(base_user_add_recipe_group)

            user = authenticate(request=request, username=username, password=password)
            if user:
                login(request, user)

            script = f"""<script>
                            window.location.href = "{reverse("catalog:recipe-home")}";
                            localStorage.clear();
                        </script>"""

            return HttpResponse(script)
    else:
        form = RegistrationForm()

    return render(request, "registration.html", {"form": form})


def unique_registration_check(request):
    if request.method == "POST":
        data = json.loads(request.body)
        field_value = data.get("field_value")
        user = User.objects.filter(
            Q(username=field_value) | Q(email=field_value)
        ).first()
        if user:
            return JsonResponse({"error_message": "Already taken"})

        return JsonResponse({})


def edit_profile_unique_check(request):
    if request.method == "POST":
        data = json.loads(request.body)
        field_value = data.get("field_value")
        check_user = User.objects.filter(
            Q(username=field_value) | Q(email=field_value)
        ).first()
        user = request.user
        if check_user and check_user.pk != user.pk:
            return JsonResponse({"error_message": "Already taken"})
        return JsonResponse({})


class LogoutView(View):
    def get(self, request):
        logout(request)
        return redirect(reverse("catalog:recipe-home"))


def func_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request=request, username=username, password=password)
        if user:
            login(request, user)
            return HttpResponse(
                json.dumps(
                    {"message": "Success", "homeUrl": reverse("catalog:recipe-home")}
                ),
                content_type="application/json",
            )
        else:
            return HttpResponse(
                json.dumps({"message": "Denied"}), content_type="application/json"
            )


class LoginRedirectPage(TemplateView):
    template_name = "login_redirect.html"


class ResetPasswordView(View):
    def get(self, request):
        return render(request, "reset_password.html")

    def post(self, request):
        email = request.POST.get("email")
        user = User.objects.filter(email=email).first()
        if user:
            number_string = [str(i) for i in range(settings.MAX_PASSWORD_NUM)]
            eng_alphabet = string.ascii_letters
            new_password = ""
            for i in range(settings.PASSWORD_ITERATION):
                new_password += random.choice(eng_alphabet)
                new_password += random.choice(number_string)
            user.set_password(new_password)
            user.save()
            send_mail(
                "Library support",
                f"""
                Your temporary password - {new_password}
                You can authorize on home page
                Home page link - {request.build_absolute_uri(reverse("catalog:recipe-home"))}
                """,
                os.getenv("EMAIL_HOST_USER"),
                [email],
                fail_silently=False,
            )
            messages.add_message(
                request,
                messages.SUCCESS,
                "Success! Check your email and sign in with new credentials.",
            )
        else:
            messages.add_message(
                request, messages.ERROR, "Warning! You entered the invalid email."
            )
        return redirect(reverse("users:reset-password"))


class HelpEmailView(View):
    form = ContactForm

    def get_form(self, data: typing.Dict = {}):
        return self.form(data)

    def get(self, request):
        return render(request, "help_email.html", {"form": self.get_form()})

    def post(self, request):
        form = self.get_form(request.POST)
        if form.is_valid():
            user_email = form.cleaned_data.get("user_email")
            subject = form.cleaned_data.get("subject")
            message = form.cleaned_data.get("message")
            send_mail(
                f"{subject}",
                f"""
                    Question : {message}\n
                    Email for answer - {user_email}
                    """,
                settings.EMAIL_HOST_USER,
                [os.getenv("ADMIN_EMAIL")],
                fail_silently=False,
            )
            messages.add_message(
                request,
                messages.SUCCESS,
                "The letter was sent, wait for a response to your mailbox",
            )
        else:
            messages.add_message(
                request,
                messages.ERROR,
                "Check your form, the fields were filled in incorrectly",
            )
        return redirect(reverse("users:help-email"))
