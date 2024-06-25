import json
from django.core.serializers import serialize
from django.template import Library


register = Library()


@register.filter(is_safe=True)
def jsonify(object):
    return json.dumps(object)
