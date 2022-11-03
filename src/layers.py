import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import folium


import requests
import urllib.request
import os
import gadm


class AdmArea():
    def __init__(self, country : str, level : int) -> None:
        self.country = country
        self.level = level
        self._get_country_data()

        
    def _get_country_data(self) -> None:
        print(f"Retrieving data for {self.country} of granularity level {self.level}")
        self.country_fc = gadm.get_data(code=self.country, level=self.level)
        print(f"Administrative areas for level {self.level}:")
        print([feat["properties"][f"NAME_{self.level}"] for feat in self.country_fc])

    def get_adm_area(self, adm_name : str) ->  None:
        self.adm_name = adm_name

        args = {f"NAME_{self.level}" : self.adm_name}
        adm_fc = self.country_fc.get(**args)
        if adm_fc:
            geometry = MultiPolygon(Polygon(shape[0]) for shape in adm_fc["geometry"]["coordinates"])
            self.geometry = geometry
        else:
            print(f"No data found for {self.adm_name}")    

class FacilityLayer():
    def __init__(self, adm_area : AdmArea) -> None:
        self.adm_area = adm_area

    def osm_populate(self, tags : dict) -> None:
        self.tags = tags
        print(f"Retrieving {tags} for {self.adm_area.adm_name} area")
        gdf = ox.geometries_from_polygon(polygon = self.adm_area.geometry, tags = self.tags)
        osmids = gdf.index.get_level_values('osmid')
        lon , lat = [], []
        for index, data in gdf.iterrows():
            if index[0] == "node":
                lon.append(data['geometry'].x)
                lat.append(data['geometry'].y)
            else:
                lon.append(data['geometry'].centroid.x)
                lat.append(data['geometry'].centroid.y)
        self.gdf = gpd.GeoDataFrame(
            index=osmids, 
            data={'lon':lon, 'lat':lat},
            geometry=gdf.geometry.values)
        print("Data loaded")      

    def plot_facilities(self) -> folium.Map:
        start_coords = list(self.adm_area.geometry.centroid.coords)[0]
        folium_map = folium.Map(
            location=tuple([start_coords[1], start_coords[0]]), width="50%", height="50%", zoom_start=6)
        try:
            for i in range(0,len(self.gdf)):
                folium.CircleMarker([self.gdf.iloc[i]['lat'], self.gdf.iloc[i]['lon']],
                        color='blue',fill=True, radius=2).add_to(folium_map)
        except:
                print("No facilities found")
        return folium_map                        

class PopulationLayer():
    def __init__(self, adm_area : AdmArea, year : int) -> None:
        self.adm_area = adm_area
        self.year = year
        self.world_pop_populate()

    def world_pop_populate(self) -> None:
        # Call worldpop API
        print(f"Retrieving population data for {self.adm_area.adm_name} area and year {self.year}")
        worldpop_url = f'https://www.worldpop.org/rest/data/pop/wpicuadj1km/?iso3={self.adm_area.country}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'} # User-Agent is required to get API response
        response = requests.get(worldpop_url, headers=headers)    

        # Extract url for data for specified year
        data = response.json()['data']
        year_data = [dataset for dataset in data if dataset['popyear'] == str(self.year)]
        url = [file for file in year_data[0]['files'] if ".zip" in file]

        # Download dataset and load into dataframe
        filehandle, _ = urllib.request.urlretrieve(url[0])
        try:
            df = pd.read_csv(filehandle,compression='zip')
        except:
            print("Something went wrong with loading the data")
            self.pop_gdf = None    
        df = df.reset_index()    
        df.columns = ['ID','lon','lat','population']
        self.pop_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    def group_population(self, nof_digits : int) -> gpd.GeoDataFrame:
        population = self.pop_gdf.copy()
        population['lon'] = population['lon'].round(nof_digits)
        population['lat'] = population['lat'].round(nof_digits)

        population = population.groupby(['lon','lat'])['population'].sum().reset_index().reset_index()
        population['population'] = population['population'].round()
        population.columns = ['ID','lon','lat','population']
        population = population.set_geometry(gpd.points_from_xy(population.lon, population.lat))
        return population

    def apply_adm_area(self) -> gpd.GeoDataFrame:
        bounds = self.adm_area.geometry.bounds
        bounded_gdf = self.pop_gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        return bounded_gdf