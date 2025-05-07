import mercantile
import requests
import pandas as pd
import random
import os

if __name__ == '__main__':
    df = pd.read_csv('./data/points/new_york/val_points.csv')
    # df = pd.read_csv('./data/points/new_york/train_points.csv')

    for index, rows in df.iterrows():
        latitude = rows['latitude']
        longitude = rows['longitude']
        point_id = rows['point_id']

        # Get the tile coordinates for the specified latitude and longitude
        tile = mercantile.tile(longitude, latitude, 16)  # Adjust the zoom level as needed

        # Construct the correct OSM tile URL with subdomain for the zoom level
        #osm_url = f"https://api.mapbox.com/v4/mapbox.satellite/{tile.z}/{tile.x}/{tile.y}@2x.jpg90?access_token=pk.eyJ1IjoidmlzaHUyNjEwIiwiYSI6ImNscmlscjJ3OTA3cWkya3MzZDBnZWU5bnYifQ.xi0Xet9du8JQlhsZiMusyw"
        osm_url = f"https://api.mapbox.com/styles/v1/cherd/cma5spbna00ab01s79ohd5ghb/tiles/512/{tile.z}/{tile.x}/{tile.y}?access_token=pk.eyJ1IjoiY2hlcmQiLCJhIjoiY204N3B3ZGNtMDdvazJrb2FvbDc0d3g5ayJ9.94JE1qkBTLJC_DxbvxxLTQ" 
        #osm_url = f"https://{ser}.tile.openstreetmap.org/{tile.z}/{tile.x}/{tile.y}.png"

        # Download the OSM tile
        output_filename = f"data/osm_images/new_york/patch_{point_id}.jpeg"

        if os.path.isfile(output_filename):
            continue

        headers = {'Accept-Language': 'en', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'}

        try:
            response = requests.get(osm_url, headers=headers)
            response.raise_for_status()
            
            with open(output_filename, 'wb') as output_file:
                output_file.write(response.content)
            
            print(f'The OSM tile has been downloaded and saved as {output_filename}')
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')