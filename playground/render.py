import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import mercantile
from shapely.geometry import mapping, Point
import skimage.draw


EXCLUDED_TAG = "administrative_boundary"


def to_pixel_coords(coords, bounds, img_size):
    west, south, east, north = bounds
    coords = np.array(coords)
    lons, lats = coords[:, 0], coords[:, 1]
    norm_x = np.clip((lons - west) / (east - west), 0, 1)
    norm_y = 1 - np.clip((lats - south) / (north - south), 0, 1)
    px = (norm_x * (img_size - 1)).astype(int)
    py = (norm_y * (img_size - 1)).astype(int)
    return px, py


def rasterize_geometry(grid, geom, tag_indices, bounds, img_size):
    def apply_polygon(polygon_coords):
        outer_ring = polygon_coords[0]
        px, py = to_pixel_coords(outer_ring, bounds, img_size)
        rr, cc = skimage.draw.polygon(py, px)
        for r, c in zip(rr, cc):
            grid[r][c].extend(tag_indices)

    def apply_line(coords):
        px, py = to_pixel_coords(coords, bounds, img_size)
        for i in range(len(px) - 1):
            rr, cc = skimage.draw.line(py[i], px[i], py[i+1], px[i+1])
            for r, c in zip(rr, cc):
                grid[r][c].extend(tag_indices)

    geom_type = geom["type"]
    coords = geom["coordinates"]

    if geom_type == "Point":
        px, py = to_pixel_coords([coords], bounds, img_size)
        for x, y in zip(px, py):
            grid[y][x].extend(tag_indices)

    elif geom_type == "LineString":
        apply_line(coords)

    elif geom_type == "MultiLineString":
        for linestring in coords:
            apply_line(linestring)

    elif geom_type == "Polygon":
        apply_polygon(coords)

    elif geom_type == "MultiPolygon":
        for polygon in coords:
            apply_polygon(polygon)

    else:
        print(f"Unsupported geometry type: {geom_type}")


def save_csv(grid, output_path, img_size):
    output_path = Path(output_path)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["row", "col", "tag_indices"])
        for row in range(img_size):
            for col in range(img_size):
                tag_ids = set(grid[row][col])
                writer.writerow([row, col, ",".join(map(str, tag_ids)) if tag_ids else ""])
    print(f"Saved CSV to {output_path}")

def save_npy(grid, output_path, img_size):
    # Initialize the grid_array with None values (or empty lists)
    grid_array = np.empty((img_size, img_size), dtype=object)

    for row in range(img_size):
        for col in range(img_size):
            # Convert list of tags to a set to ensure uniqueness
            grid_array[row, col] = list(set(grid[row][col]))  # Convert to set to remove duplicates
    
    # Save to .npy file
    np.save(output_path, grid_array)
    print(f"Saved .npy to {output_path}")

def build_tag_vocab(json_dir, point_ids):
    tag_set = set()
    for point_id in point_ids:
        json_path = Path(json_dir) / f"{point_id}.json"
        if not json_path.exists():
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for geom, tags in data.values():
            filtered_tags = [tag for tag in tags if tag != EXCLUDED_TAG]
            tag_set.update(filtered_tags)
    tag_list = sorted(tag_set)
    tag_to_index = {tag: i for i, tag in enumerate(tag_list)}
    return tag_to_index


def generate_pixel_tag_grid(json_path, point_id, POINTS_DF, tag_to_index, tile_z=16, img_size=512):
    lon, lat = POINTS_DF.loc[POINTS_DF['point_id'] == point_id, ['longitude', 'latitude']].values[0]
    tile = mercantile.tile(lon, lat, tile_z)
    bounds = mercantile.bounds(tile)

    with open(json_path, 'r') as f:
        data = json.load(f)

    grid = [[[] for _ in range(img_size)] for _ in range(img_size)]
    for key in data:
        geom, tags = data[key]
        filtered = [t for t in tags if t != EXCLUDED_TAG and t in tag_to_index]
        if not filtered:
            continue
        indices = [tag_to_index[t] for t in filtered]
        rasterize_geometry(grid, geom, indices, bounds, img_size)
    return grid, bounds

def save_tag_vocab_json(tag_vocab, output_path):
    with open(output_path, 'w') as f:
        json.dump(tag_vocab, f)
    print(f"Saved tag vocab to {output_path}")


def main(POINTS_DF, json_dir, output_dir, point_ids=None, save_csv_flag=True, save_npy_flag=True, save_vocab_flag=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    point_ids = point_ids if point_ids is not None else POINTS_DF['point_id'].unique()
    tag_to_index = build_tag_vocab(json_dir, point_ids)
    if save_vocab_flag:
        vocab_output_path = output_dir / "tag_vocab.json"
        save_tag_vocab_json(tag_to_index, vocab_output_path)

    for point_id in point_ids:
        json_path = Path(json_dir) / f"{point_id}.json"
        output_path = output_dir / f"{point_id}.csv"
        npy_output_path = output_dir / f"{point_id}.npy"

        if not json_path.exists():
            print(f"Skipping {point_id}: JSON not found.")
            continue

        grid, bounds = generate_pixel_tag_grid(json_path, point_id, POINTS_DF, tag_to_index)

        if save_csv_flag:
            save_csv(grid, output_path, img_size=512)
        if save_npy_flag:
            save_npy(grid, npy_output_path, img_size=512)
        
if __name__ == "__main__":
    POINTS_DF = pd.read_csv("/data/cher/rendersynth/data/val_points.csv")
    NYC_TAGS_PATH = '/data/cher/rendersynth/data/val/vector'
    output_dir = "/data/cher/rendersynth/playground/outputs/pixel_tags"

    main(POINTS_DF, 
         json_dir=NYC_TAGS_PATH, 
         output_dir=output_dir, 
         save_csv_flag=True)
