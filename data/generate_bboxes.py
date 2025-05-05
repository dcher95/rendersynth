import pandas as pd
from shapely.geometry import box, mapping
import geojson
import mercantile
import os
from dotenv import load_dotenv

def generate_bboxes(input_csv, output_dir, combined_output_path):
    if os.path.exists(combined_output_path):
        print(f"Combined output already exists at {combined_output_path}. Skipping generation.")
        return
    
    points = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    combined_features = []

    for _, row in points.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        point_id = row['point_id']

        output_file = os.path.join(output_dir, f"{point_id}.geojson")
        if os.path.exists(output_file):
            print(f"Skipping existing file: {output_file}")
            # Optionally skip reading and adding it to combined
            continue

        tile = mercantile.tile(longitude, latitude, 16)
        bbox = mercantile.bounds(tile)
        bbox_geom = box(bbox.west, bbox.south, bbox.east, bbox.north)

        feature = {
            "type": "Feature",
            "geometry": mapping(bbox_geom),
            "properties": {"point_id": point_id}
        }

        # Save individual file
        with open(output_file, 'w') as f:
            geojson.dump({"type": "FeatureCollection", "features": [feature]}, f)

        # Add to combined list
        combined_features.append(feature)

    # Save combined file
    with open(combined_output_path, 'w') as f:
        geojson.dump({"type": "FeatureCollection", "features": combined_features}, f)

    print(f"Saved {len(combined_features)} individual files and combined GeoJSON to {combined_output_path}.")

if __name__ == "__main__":
    load_dotenv("config.env")

    input_csv = os.getenv("POINTS_CSV")
    output_dir = os.getenv("BOUNDING_BOXES_DIR")
    combined_output_path = os.getenv("COMBINED_BBOX_OUTPUT_PATH")

    generate_bboxes(input_csv, output_dir, combined_output_path)
