from django.urls import path
from .views import (LandingPageView,
                    main_func_temp)

app_name = 'cpai'

urlpatterns = [
    path('', LandingPageView.as_view(), name='test_view'),
    path('main/', main_func_temp, name='main'),
]