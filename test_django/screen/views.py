from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import AddressForm
from .models import Address
from .fire_model import FireModel
import logging
import pandas as pd
import requests
import math
# Create your views here.

risk_df = pd.read_csv("screen/wildfire_risk.csv")

logger = logging.getLogger(__name__)

def screen(request): 
    if request.method == 'POST': 
        #update location button
        if 'update_location' in request.POST: 
            request.session.pop('county', None)
            request.session.pop('lat', None)
            request.session.pop('lon', None)
            request.session.pop('acres_burned', None)
            request.session.pop('risk_percent', None)
            request.session['has_address'] = False
            print("location cleared")
            return redirect('screen')
        
        address = request.POST.get('county', '')
        logger.info("received address input -city {city}, Address: {county}")

        #initialize 
        fireModel = FireModel()
        

        lat, lon = geocode_address(address)

        risk_percent = risk_factor1(lat, lon)
        acres_burned = math.ceil(fireModel.predict(lat,lon))

        request.session.update({ 
            'county': address, 
            'lat': lat, 
            'lon': lon,
            'risk_percent': risk_percent,
            'acres_burned': acres_burned,
            'has_address': True


        })
        return redirect('screen')

    context = { 
        'has_address': request.session.get('has_address', False),
        'county': request.session.get('county', 
        ''),
        'lat': request.session.get('lat', ''),
        'lon': request.session.get('lon', ''),
        'risk_percent': request.session.get('risk_percent', ''), 
        'acres_burned': request.session.get('acres_burned', '')
    }
    template = loader.get_template('index.html')
    return HttpResponse(template.render(context, request))

def acres(): 
    return 100

def geocode_address(address): 
    api_key = "AIzaSyDXNho9hSiTCsVRDwShgVSzDI0VbUCH7wA"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    response = requests.get(url, params=params)
    results = response.json()["results"]

    if results: 
        location = results[0]["geometry"]["location"]
        return location["lat"], location["lng"]
    else: 
        return None

def risk_factor1(lat, lon): 
    api_key = "AIzaSyDXNho9hSiTCsVRDwShgVSzDI0VbUCH7wA"
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    response = requests.get(url)
    results = response.json().get("results", [])
    city = None
    if results: 
        for component in results[0]["address_components"]:
            if "locality" in component["types"]:
                city = component["long_name"]
                break
    
    city = city.strip().lower()
    risk_df['City_normalized'] = risk_df['City'].str.strip().str.lower()

    match = risk_df[risk_df['City_normalized'] == city]
    
    if not match.empty: 
        risk_percent = match.iloc[0]['Risk Rank']
        return(risk_percent)






def address_view(request):
    if request.method == 'POST':
        form = AddressForm(request.POST)
        if form.is_valid():
            address = form.save()
            request.session.update({
                'user_street': address.street_address,
                'user_city': address.city,
                'user_state': address.state,
                'user_zip': address.zip_code,
                'user_country': address.country
            })
            return redirect('address_success')
    else:
        form = AddressForm()
    
    return render(request, 'address_form.html', {'form': form})

def address_success(request):
    context = {
        'address': {
            'street': request.session.get('user_street'),
            'city': request.session.get('user_city'),
            'state': request.session.get('user_state'),
            'zip': request.session.get('user_zip'),
            'country': request.session.get('user_country')
        }
    }
    return render(request, 'address_success.html', context)