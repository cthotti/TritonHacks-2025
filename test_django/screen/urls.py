from django.urls import path 
from . import views
from . import models

urlpatterns = [
    path('screen/', views.screen, name='screen'),
    path('screen/address', views.address_view, name='address_form'), 
    path('screen/success/', views.address_success, name='address_success'),
]