import pandas as pd
from geopy import Nominatim
import folium




HomeState_data=pd.read_csv("HomeState_data.csv")

locator = Nominatim(user_agent="Rishitha")


lat_and_long=list(zip(HomeState_data['Latitude'],HomeState_data['Longitude']))


HomeState_map = folium.Map(location=lat_and_long[0] ,tiles="openstreetmap",zoom_start=7)

i=0
for pair in lat_and_long:

          folium.Marker(location=pair ,popup=HomeState_data.iloc[i]['Pincode']).add_to(HomeState_map)
          i+=1
          
HomeState_map.save("HomeState_map.html")


