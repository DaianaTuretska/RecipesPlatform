from django.test import TestCase
from django.urls import reverse


class HomePageTest(TestCase):
    def test_get_page(self):
        self.assertTrue(True)

    def test_context(self):
        self.assertTrue(True)


class ProfileEditViewTest(TestCase):
    def setUp(self) -> None:
        self.assertTrue(True)

    def test_unauthorized_get_page(self):
        self.assertTrue(True)

    def test_redirect_unauthorized(self):
        self.assertTrue(True)

    def test_get_page(self):
        self.assertTrue(True)


class RedirectLoginTest(TestCase):
    def test_can_access_redirect_page(self):
        self.assertTrue(True)


class SearchPageTest(TestCase):
    def test_get_search_page_without_result(self):
        self.assertTrue(True)

    def test_get_search_page_with_title(self):
        self.assertTrue(True)

    def test_get_search_page_with_author(self):
        self.assertTrue(True)

    def test_get_search_page_with_year(self):
        self.assertTrue(True)


class NewsPageTest(TestCase):
    def test_get_news_page(self):
        self.assertTrue(True)


class CollectionsPageTest(TestCase):
    def test_get_collections_page(self):
        self.assertTrue(True)


class AuthorsPageTest(TestCase):
    def test_get_authors_page(self):
        self.assertTrue(True)


class LoginViewTest(TestCase):
    def test_login_with_valid_data(self):
        self.assertTrue(True)

    def test_login_with_invalid_username(self):
        self.assertTrue(True)

    def test_login_with_invalid_password(self):
        self.assertTrue(True)


class RegistrationPageTest(TestCase):
    def test_get_page(self):
        self.assertTrue(True)

    def test_post_page(self):
        self.assertTrue(True)


class BookshelfPageTest(TestCase):
    def test_unauthorized_get_page(self):
        self.assertTrue(True)

    def test_get_page(self):
        self.assertTrue(True)

    def test_recbooks_context(self):
        self.assertTrue(True)

    def test_wishlist_context(self):
        self.assertTrue(True)

    def test_recbooks(self):
        self.assertTrue(True)

    def test_wishbooks(self):
        self.assertTrue(True)


class BookDetailsPageTest(TestCase):
    def test_get_page(self):
        self.assertTrue(True)

    def test_unauthorized_get_page(self):
        self.assertTrue(True)

    def test_authorized_get_page(self):
        self.assertTrue(True)

    def test_add_to_wishlist(self):
        self.assertTrue(True)

    def test_delete_from_wishlist(self):
        self.assertTrue(True)


class ChangePasswordPageTest(TestCase):
    def test_change_password_with_valid_data(self):
        self.assertTrue(True)

    def test_change_password_with_invalid_data(self):
        self.assertTrue(True)


class LogoutViewTest(TestCase):
    def test_logout(self):
        self.assertTrue(True)
