import json
from django.views import View
from django.http import JsonResponse
from django.core.serializers.json import DjangoJSONEncoder
from .distilbert.search import search, format_recipe
from .models import Message


# Create your views here.
class VirtualAssistantView(View):
    def post(self, request):
        error = ""

        if not request.user.is_authenticated:
            return JsonResponse({"message": "Not authorized"})

        body = json.loads(request.body)
        query = body.get("message")

        if query:
            results = search(query, top_k=1)
            result = results.iloc[0]

            name = result.recipe_name
            ingredients = result.ingredients
            directions = result.directions
            total_time = result.total_minutes

            Message.objects.create(
                query=query,
                recipe_name=name,
                ingredients=ingredients,
                directions=directions,
                total_time=total_time,
                author=request.user,
            )

        else:
            error = "The comment field should not be blank."
            name = ""
            ingredients = ""
            directions = ""
            total_time = ""
            response = ""

        return JsonResponse(
            {
                "query": query,
                "name": name,
                "ingredients": ingredients,
                "directions": directions,
                "total_time": total_time,
                "error": error,
            }
        )
