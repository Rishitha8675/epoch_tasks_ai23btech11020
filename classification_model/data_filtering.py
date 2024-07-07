import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


data = pd.read_csv("clustering_data.csv", low_memory=False)


unfiltered_data = data[data['StateName'] == 'ANDHRA PRADESH']


unfiltered_data.loc[:, 'Latitude'] = pd.to_numeric(unfiltered_data['Latitude'], errors='coerce')
unfiltered_data.loc[:, 'Longitude'] = pd.to_numeric(unfiltered_data['Longitude'], errors='coerce')


unfiltered_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
unfiltered_data.drop_duplicates()

plt.scatter(unfiltered_data['Latitude'], unfiltered_data['Longitude'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('UnFiltered data with many datapoints those are not actually present in the state')
plt.show()


andhra_pradesh_geojson = gpd.read_file("andhra_pradesh.geojson")
geometry = [Point(xy) for xy in zip(unfiltered_data['Longitude'], unfiltered_data['Latitude'])]
geo_df = gpd.GeoDataFrame(unfiltered_data, geometry=geometry)
HomeState_data = gpd.sjoin(geo_df, andhra_pradesh_geojson, how="inner")


fig, ax = plt.subplots(figsize=(10, 10))
andhra_pradesh_geojson.plot(ax=ax, color='black', edgecolor='blue')
HomeState_data.plot(ax=ax, color='red', markersize=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Filtered Points within Andhra Pradesh')
plt.show()



HomeState_data=HomeState_data.drop(columns=['geometry','index_right', 'STNAME', 'STCODE11','STNAME_SH', 'Shape_Length', 'Shape_Area', 'OBJECTID', 'State_LGD'])
print(HomeState_data)

plt.scatter(HomeState_data['Latitude'], HomeState_data['Longitude'])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

HomeState_data.to_csv('HomeState_data.csv')

