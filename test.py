import geopandas as gpd
from geopandas.tools import clip
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
import os




# min/max latitude and longitude for southern california
minx, maxx = -122, -114.0
miny, maxy = 32.0, 36

# take in CA Counties shape files
counties = gpd.read_file('ca_counties/CA_Counties.shp')
"""socal_counties = counties[counties['NAME'].isin([
    'San Diego', 'Orange', 'Los Angeles', 'Riverside', 'San Bernardino',
    'Ventura', 'Imperial', 'Santa Barbara', 'San Luis Obispo', 'Kern'
])]"""


"""# creates geodataframe for socal, converts to epsg 3310 (projection)
bbox = box(minx,miny,maxx,maxy)
gdf_bounds = gpd.GeoDataFrame({'geometry':[bbox]},crs='EPSG:4326')
gdf_bounds_proj = gdf_bounds.to_crs('EPSG:3310')"""



"""# creates the shapes for the grid cells with grid size and the bounds of socal
minx_proj, miny_proj, maxx_proj, maxy_proj = gdf_bounds_proj.total_bounds
grid_size = 10_000
grid_cells = []
x_range = np.arange(minx_proj, maxx_proj, grid_size)
y_range = np.arange(miny_proj, maxy_proj, grid_size)
for x in x_range:
    for y in y_range:
        cell = box(x,y,x+grid_size,y+grid_size)
        grid_cells.append(cell)

# creates geodata frame for the grid cells, converts back to epsg 4326
grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs='EPSG:3310')
grid_gdf = grid_gdf.to_crs('EPSG:4326')"""

# clips the grid to stop all grid squares that are not in the socal boundaries
counties = counties.to_crs('EPSG:4326')
"""grid_clipped = gpd.clip(grid_gdf, socal_counties)"""

"""# plot results
grid_gdf.plot(edgecolor='gray', facecolor='none')
plt.title('100 km Grid Over Southern California')
plt.show()"""

"""grid_clipped['rep_point'] = grid_clipped.geometry.representative_point()"""

"""with rasterio.open('temperatures/PRISM_ppt_30yr_normal_4kmM4_annual_bil.bil') as src:
    print(src.count)  # number of bands
    print(src.width, src.height)  # raster dimensions

    # Read the first band
    data = src.read(1)
    for point in grid_clipped['rep_point']:
        print(point)
        # Sample at one or more coordinates (lon, lat)
        coords = [(point.x,point.y)]
        values = [val[0] for val in src.sample(coords)]
        print(values)"""


"""{# Example: San Diego coordinates
lon, lat = -117.1611, 32.7157
temp = query_temperature(lon, lat)
print(f"Temperature at ({lat}, {lon}): {temp}")}"""

fig, ax = plt.subplots(figsize=(10, 10))
counties.boundary.plot(ax=ax, color='black', linewidth=1)
#grid_clipped.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=0.8)
plt.title("100 km Grid Clipped to Southern California")
plt.show()

