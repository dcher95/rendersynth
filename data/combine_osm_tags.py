import geopandas as gpd
import os
import pandas as pd
import json
from dotenv import load_dotenv

# Function to combine all OSM tag geojson files for a specific bbox_id
def combine_osm_tags(bbox_id, geojson_files_path):
    combined_gdf = None
    tag_stats = {}

    bbox_dir = os.path.join(geojson_files_path, bbox_id)
    if not os.path.exists(bbox_dir):
        print(f"Directory for {bbox_id} not found, skipping.")
        return None, tag_stats

    for tag_file in os.listdir(bbox_dir):
        if tag_file.endswith(".geojson"):
            tag_name = tag_file.replace(".geojson", "")
            geojson_file = os.path.join(bbox_dir, tag_file)

            try:
                gdf = gpd.read_file(geojson_file)
                feature_count = len(gdf)

                if feature_count > 0:
                    gdf["source_tag"] = tag_name
                    tag_stats[tag_name] = feature_count

                    if combined_gdf is None:
                        combined_gdf = gdf
                    else:
                        combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

            except Exception as e:
                print(f"Error reading {geojson_file}: {e}")
    return combined_gdf, tag_stats


def condense(d_gdf):
    condensed = {}

    for _, row in d_gdf.iterrows():
        name = row.get("id")

        geometry = row.geometry.__geo_interface__
        source_tag = row.get("source_tag")

        # Try to extract a subtag from column names or values
        sub_tag = None
        for col in d_gdf.columns:
            if col not in ["geometry", "source_tag", "name"] and pd.notnull(row.get(col)):
                if col.startswith(f"{source_tag}:") or col == source_tag:
                    sub_tag = col.replace(f"{source_tag}:", "") if ":" in col else row[col]
                    break

        tag_string = f"{source_tag}_{sub_tag}" if sub_tag else source_tag
        condensed[str(name)] = [geometry, [tag_string]]

    return condensed


def main():
    load_dotenv("config.env")
    
    clipped_geojson_files_path = os.getenv("CLIPPED_BBOX_GEOJSON_DIR")
    output_vector_dir = os.getenv("OUTPUT_VECTOR_DIR")
    output_geojson_dir = os.getenv("OUTPUT_GEOJSON_DIR")

    os.makedirs(output_geojson_dir, exist_ok=True)
    os.makedirs(output_vector_dir, exist_ok=True)

    summary_records = []

    bbox_ids = [d for d in os.listdir(clipped_geojson_files_path) if os.path.isdir(os.path.join(clipped_geojson_files_path, d))]

    for bbox_id in bbox_ids:
        print(f"Processing bbox {bbox_id}...")

        combined_gdf, tag_stats = combine_osm_tags(bbox_id, clipped_geojson_files_path)

        if combined_gdf is not None:
            output_file = os.path.join(output_geojson_dir, f"{bbox_id}_combined_osm.geojson")
            combined_gdf.to_file(output_file, driver='GeoJSON')
            print(f"Saved combined data for {bbox_id} to {output_file}.")

            # Also save a more condensed vector json version
            condensed_output_file = os.path.join(output_vector_dir, f"{bbox_id}.json")
            condensed_output = condense(d_gdf = combined_gdf.copy())
            with open(condensed_output_file, "w") as f:
                json.dump(condensed_output, f)

            for tag, count in tag_stats.items():
                summary_records.append({"bbox_id": bbox_id, "tag": tag, "feature_count": count})
        else:
            print(f"No data found for {bbox_id}, skipping.")

    # Write summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_vector_dir, "bbox_tag_summary.csv")
    summary_df.sort_values(by=["bbox_id", "tag"], inplace=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nTag summary written to {summary_csv}.")

if __name__ == "__main__":
    main()
