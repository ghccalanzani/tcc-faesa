from django.contrib import admin
from django.urls import path, include
from rpa_interface import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('rpa_interface.urls')),  # Rota principal
]
