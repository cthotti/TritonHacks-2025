from django.db import models
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader


def test_python(request): 
    output = "test output"
    return render(request, 'myfirst.html', {'test': output})