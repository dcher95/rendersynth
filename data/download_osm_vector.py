import osmnx as ox
import pandas as pd
import mercantile
import json
from shapely.geometry import shape
import geojson
import os
import warnings
import timeout_decorator
import time 

warnings.filterwarnings("ignore")

TOP_LEVEL_OSM_TAGS = [
    "aerialway",
    "aeroway",
    "amenity",
    "barrier",
    "boundary",
    "building",
    "craft",
    "emergency",
    "geological",
    "healthcare",
    "highway",
    "historic",
    "landuse",
    "leisure",
    "man_made",
    "military",
    "natural",
    "office",
    "place",
    "power",
    "public_transport",
    "railway",
    "route",
    "shop",
    "sport",
    "telecom",
    "tourism",
    "water",
    "waterway",
]

# @timeout_decorator.timeout(100)
def get_osm_data(latitude, longitude, point_id):
    start_time = time.time()  # Start timing
    
    d = {}

    tile = mercantile.tile(longitude, latitude, 16)
    bbox = mercantile.bounds(tile)

    print(f"Bounding box for point {point_id}: {bbox}")  # Debug print

    prompt = {}

    try:
        gdf = ox.features_from_bbox(
            bbox=(bbox.north, bbox.south, bbox.east, bbox.west),
            tags={k: True for k in TOP_LEVEL_OSM_TAGS}
        )
        if gdf.empty:
            print(f"No data found for point {point_id} in bbox: {bbox}")  # Debug print

        gdf.drop(columns=gdf.columns.intersection(['nodes', 'ways']), inplace=True)

        for key in TOP_LEVEL_OSM_TAGS:
            if key in gdf.columns:
                l = gdf[gdf[key].notna()]
                for i, row in l.iterrows():
                    if str(row.name) not in prompt:
                        prompt[str(row.name)] = (row['geometry'], [str(key)+"_"+str(row[key])])
                    else:
                        prompt[str(row.name)][1].append(str(key)+"_"+str(row[key]))
    except Exception as e:
        print(f"Error processing point {point_id}: {e}")  # Debug print

    d[point_id] = prompt

    end_time = time.time()  # End timing
    elapsed = end_time - start_time
    print(f"Time taken for point {point_id}: {elapsed:.2f} seconds")
    
    return d



def main():
    data = pd.read_csv('./data/points/new_york/val_points.csv').head(1)

    os.makedirs('data/vector/new_york/val_vector', exist_ok=True)

    for index, row in data.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        point_id = row['point_id']

        print(f"Processing point {point_id}...")
        d = get_osm_data(latitude, longitude, point_id)

        with open(f'data/vector/new_york/val_vector/{point_id}_osm.json', 'w') as f:
            geojson.dump(d, f)

if __name__ == "__main__":
    main()

