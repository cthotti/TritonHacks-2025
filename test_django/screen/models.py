from django.db import models
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

class Address(models.Model):
    street_address = models.CharField(max_length=100)
    city = models.CharField(max_length=50)
    state = models.CharField(max_length=50)
    zip_code = models.CharField(max_length=10)
    country = models.CharField(max_length=50)
    lat = models.FloatField()
    lon = models.FloatField()
    created_at = models.DateTimeField

    def __str__(self): 
        return f"{self.street_address}, {self.city}"
