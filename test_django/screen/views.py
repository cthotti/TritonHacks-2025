from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
# Create your views here.

def screen(request): 
    template = loader.get_template('myfirst.html')
    return HttpResponse(template.render())

def test_python(request): 
    return render(request, 'myfirst.html', {'test_python': "THIS IS A TEST"})