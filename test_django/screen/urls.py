from django.urls import path 
from . import views
from . import models

urlpatterns = [
    path('screen/', views.screen, name='screen'),
    path('screen/test_python/', views.test_python, name='test_python'),
]