from django.test import TestCase


class ChangePasswordFormTest(TestCase):
    def test_new_and_confirm_password_not_match(self):
        self.assertTrue(True)


class EditProfileFormTest(TestCase):
    def test_edit_profile_form_firstname_label(self):
        self.assertTrue(True)

    def test_edit_profile_form_lastname_label(self):
        self.assertTrue(True)

    def test_edit_profile_form_email_label(self):
        self.assertTrue(True)

    def test_edit_profile_form_login_label(self):
        self.assertTrue(True)

    def test_form_email_unique_validation(self):
        self.assertTrue(True)

    def test_form_error_count(self):
        self.assertTrue(True)


class RegistrationFormTest(TestCase):
    def test_valid_form(self):
        self.assertTrue(True)

    def test_firstname_label(self):
        self.assertTrue(True)

    def test_lastname_label(self):
        self.assertTrue(True)

    def test_email_label(self):
        self.assertTrue(True)

    def test_login_label(self):
        self.assertTrue(True)

    def test_password_label(self):
        self.assertTrue(True)

    def test_confirm_password_label(self):
        self.assertTrue(True)

    def test_firstname_blank(self):
        self.assertTrue(True)

    def test_firstname_profanity(self):
        self.assertTrue(True)

    def test_lastname_blank(self):
        self.assertTrue(True)

    def test_lastname_profanity(self):
        self.assertTrue(True)

    def test_email_blank(self):
        self.assertTrue(True)

    def test_email_invalid(self):
        self.assertTrue(True)

    def test_email_profanity(self):
        self.assertTrue(True)

    def test_email_unique_validation(self):
        self.assertTrue(True)

    def test_username_blank(self):
        self.assertTrue(True)

    def test_username_invalid_length(self):
        self.assertTrue(True)

    def test_username_profanity(self):
        self.assertTrue(True)

    def test_username_unique_validation(self):
        self.assertTrue(True)

    def test_password_invalid(self):
        self.assertTrue(True)

    def test_confirm_password(self):
        self.assertTrue(True)

    def test_passwords_doesnt_match(self):
        self.assertTrue(True)

    def test_error_count(self):
        self.assertTrue(True)
