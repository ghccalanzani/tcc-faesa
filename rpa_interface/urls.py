from django.urls import path
from . import views

urlpatterns = [
    path('', views.classificar_documentos_view, name='classificador'),
]