import geopandas as gpd
import pandas as pd
from geopandas.tools import clip
from shapely.geometry import box
import matplotlib.pyplot as plt


minx, maxx = -122, -114.0
miny, maxy = 32.0, 36

counties = gpd.read_file('ca_counties/CA_Counties.shp')
df = pd.read_csv('county_risks.csv')
df = df[['COUNTY','RISK_RATNG']]
counties['risk_rating'] = df['RISK_RATNG']

fig, ax = plt.subplots(figsize=(6, 6))
print(fig,ax)
counties.plot(column='risk_rating',legend=True,ax=ax, linewidth=1,cmap='OrRd_r')
plt.title("Risk Rating for Counties in California")
plt.show()
