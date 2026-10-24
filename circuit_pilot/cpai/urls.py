from django.urls import path
from .views import test_view

app_name = 'cpai'

urlpatterns = [
    path('', test_view, name='test_view'),
]