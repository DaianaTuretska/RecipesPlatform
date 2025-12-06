from django.urls import path

from . import views


app_name = "virtual_assistant"

urlpatterns = [
    path(
        "virtual-assistant/",
        views.VirtualAssistantView.as_view(),
        name="virtual-assistant",
    ),
]
