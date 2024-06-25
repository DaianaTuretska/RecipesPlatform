from django import forms

from ckeditor.fields import CKEditorWidget

from . import models


class RecipeReviewForm(forms.Form):
    comment = forms.CharField(
        widget=CKEditorWidget(
            attrs={
                "class": "form-control",
                "placeholder": "Your comment...",
                "id": "exampleFormControlTextarea1",
                "name": "text-comment",
                "style": "width: 100%",
            }
        )
    )
