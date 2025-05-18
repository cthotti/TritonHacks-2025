from django.test import TestCase

# Create your tests here.
import requests
import pandas as pd
risk_df = pd.read_csv("screen/wildfire_risk.csv")

def geocode_address(address): 
    api_key = "AIzaSyDXNho9hSiTCsVRDwShgVSzDI0VbUCH7wA"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }
    lat, lon = None, None
    response = requests.get(url, params=params)
    results = response.json()["results"]

    if results: 
        location = results[0]["geometry"]["location"]
        lat = location["lat"]
        lon = location['lng']
    
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
    



def get_city_from_latlon(lat, lon): 
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






import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

shapefile_path = "screen/CA_Counties/CA_Counties.shp"
gdf = gpd.read_file(shapefile_path)
print(gdf.head())
gdf.plot(figsize=(10, 10), edgecolor='black')
plt.title("California Counties")
plt.show()


#openweatherapi:  7b98b7e11c8589fbbc2f3f74ec95803b
#(32.621458, -116.988865)