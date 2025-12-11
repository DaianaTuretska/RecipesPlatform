import json
from django.views import View
from django.http import JsonResponse
from .llm.generate_recipe import ask
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
            print("Incoming prompt:", query)
            response = ask(query, top_k=10)

            Message.objects.create(
                query=query,
                recipe_name=response["name"],
                ingredients=response["ingredients"],
                directions=response["directions"],
                total_time=response["time"],
                author=request.user,
            )

        else:
            error = "The comment field should not be blank."
            response = {
                "name": "",
                "ingredients": "",
                "directions": "",
                "total_time": "",
                "closing_phrase": "",
            }

        return JsonResponse(
            {
                "query": query,
                "error": error,
                **response,
            }
        )
