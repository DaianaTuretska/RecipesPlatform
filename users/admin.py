from django.contrib import admin

# Register your models here.
from . import models
from . import choices


@admin.register(models.User)
class UserAdmin(admin.ModelAdmin):
    list_display = ["id", "username", "firstname", "lastname", "date_joined"]
    search_fields = ["username", "firstname", "lastname"]
    list_filter = ["date_joined"]
    ordering = ("-date_joined",)

    def save_model(self, request, obj, change, form):
        if obj.pk:
            orig_obj = models.User.objects.get(pk=obj.pk)
            if obj.password != orig_obj.password:
                obj.set_password(obj.password)
        else:
            obj.set_password(obj.password)
        obj.save()

    def has_add_permission(self, request):
        return request.user.is_superuser or request.user.role == choices.Role.ADMIN

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser or request.user.role == choices.Role.ADMIN

    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser or request.user.role == choices.Role.ADMIN


admin.site.unregister(models.User)
admin.site.register(models.User, UserAdmin)
