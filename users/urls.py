from django.urls import path

from . import views


app_name = "users"

urlpatterns = [
    path("registration/", views.registration, name="registration"),
    path(
        "registration_unique_validation/",
        views.unique_registration_check,
        name="library-registration-validate",
    ),
    path(
        "edit_profile_unique_validation/",
        views.edit_profile_unique_check,
        name="edit-profile-validate",
    ),
    path(
        "profile_details/",
        views.ProfileDetailsView.as_view(),
        name="profile-details",
    ),
    path(
        "profile_saved_recipes/",
        views.profile_saved_recipes,
        name="profile-saved-recipes",
    ),
    path("profile_my_recipes/", views.profile_my_recipes, name="profile-my-recipes"),
    path("profile_edit/", views.ProfileEditView.as_view(), name="profile-edit"),
    path("change_password/", views.change_password, name="change-password"),
    path("func_login", views.func_login, name="func-login"),
    path(
        "login_redirect_page/",
        views.LoginRedirectPage.as_view(),
        name="login-redirect-page",
    ),
    path("logout/", views.LogoutView.as_view(), name="logout"),
    path("reset_password/", views.ResetPasswordView.as_view(), name="reset-password"),
    path("help/", views.HelpEmailView.as_view(), name="help-email"),
]
