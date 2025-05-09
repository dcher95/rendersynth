import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from shapely.geometry import Point, shape, mapping, box, Polygon, LineString
from rasterio.features import rasterize
from affine import Affine
import mercantile
from PIL import Image
from tqdm import tqdm
import cv2
from collections import defaultdict
from shapely.geometry import mapping
import mercantile
import matplotlib.patches as mpatches
import random
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- 1. TAG SEMANTICS & DISTRIBUTION ---

def extract_tags_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return [tag for _, (_, tag_list) in data.items() for tag in tag_list]

def analyze_tag_distribution(json_dir):
    top_tag_counter = Counter()
    tag_hierarchy = defaultdict(Counter)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    for file in json_files:
        tags = extract_tags_from_json(os.path.join(json_dir, file))
        for tag in tags:
            if "_" in tag:
                top, sub = tag.split("_", 1)
                top_tag_counter[top] += 1
                tag_hierarchy[top][sub] += 1
            else:
                top_tag_counter[tag] += 1
                tag_hierarchy[tag][""] += 1

    return top_tag_counter, tag_hierarchy

def plot_top_tags(counter, title="Top-level OSM Tags", top_n=20):
    top = counter.most_common(top_n)
    tags, counts = zip(*top)
    plt.figure(figsize=(10, 6))
    plt.barh(tags, counts)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_sunburst(tag_tree):
    rows = []
    for top_tag, sub_counter in tag_tree.items():
        for sub_tag, count in sub_counter.items():
            rows.append((top_tag, sub_tag, count))

    df = pd.DataFrame(rows, columns=["TopTag", "SubTag", "Count"])
    fig = px.sunburst(df, path=["TopTag", "SubTag"], values="Count", title="Tag Hierarchy (Sunburst)")
    fig.show()

def plot_treemap(tag_tree):
    rows = []
    for top_tag, sub_counter in tag_tree.items():
        for sub_tag, count in sub_counter.items():
            label = f"{top_tag}/{sub_tag}" if sub_tag else top_tag
            rows.append((top_tag, label, count))

    df = pd.DataFrame(rows, columns=["TopTag", "Label", "Count"])
    fig = px.treemap(df, path=["TopTag", "Label"], values="Count", title="Tag Tree Map")
    fig.show()

def get_combined_tags(tag_tree):
    combined_tags = []
    for top_tag, counter in tag_tree.items():
        for sub_tag in counter:
            if sub_tag:  # Skip empty sub-tags
                combined_tags.append(f"{top_tag}_{sub_tag}")

    # Optionally sort or deduplicate
    combined_tags = sorted(set(combined_tags))

    return combined_tags

# --- 2. SPATIAL COVERAGE & DENSITY ---


TILE_SIZE = 512  # or 256 depending on your raster

def compute_density_from_vector_file(json_path, tile_bbox):
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = []
    for obj in data.values():
        geo = obj[0]
        geom = shape(geo)
        if not geom.is_valid:
            continue
        shapes.append((geom, 1))  # 1 = value for mask

    if not shapes:
        return 0.0, 1.0, np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    # Create transform from bounding box
    transform = Affine.translation(tile_bbox.west, tile_bbox.north) * Affine.scale(
        (tile_bbox.east - tile_bbox.west) / TILE_SIZE,
        -(tile_bbox.north - tile_bbox.south) / TILE_SIZE  # NOTE: negative scale for Y
    )

    mask = rasterize(
        [(mapping(geom), value) for geom, value in shapes],
        out_shape=(TILE_SIZE, TILE_SIZE),
        transform=transform,
        fill=0,
        all_touched=True  # rasterize full coverage
    )

    density = mask.sum() / (TILE_SIZE * TILE_SIZE)
    sparsity = 1 - density
    return density, sparsity, mask

def summarize_vector_dir(json_dir, points_df, task='density'):
    metrics = []
    agg_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint32)
    total_files = 0

    for _, row in tqdm(points_df.iterrows(), total=points_df.shape[0]):
        point_id = row['point_id']
        lon, lat = row['longitude'], row['latitude']
        vector_path = os.path.join(json_dir, f"{point_id}.json")
        if not os.path.exists(vector_path):
            continue

        tile = mercantile.tile(lon, lat, 16)
        bbox = mercantile.bounds(tile)
        if task == 'density':
            density, sparsity, mask = compute_density_from_vector_file(vector_path, bbox)
            metrics.append({
                "point_id": point_id,
                "density": density,
                "sparsity": sparsity
            })

            agg_mask += mask
            total_files += 1

            # Normalize to get % coverage per pixel
            coverage = agg_mask.astype(np.float32) / max(total_files, 1)
        
        if task == 'tag_density':
            avg_density, tag_count_map = compute_all_tags_density(vector_path, bbox)
            metrics.append({
                "point_id": point_id,
                "avg_density": avg_density
            })
            agg_mask += tag_count_map
            total_files += 1
            coverage = agg_mask.astype(np.float32) / max(total_files, 1)

    return pd.DataFrame(metrics), coverage

def plot_image_and_mask(point_id, mask_dir, image_dir, points_df):
    vector_path = os.path.join(mask_dir, f"{point_id}.json")
    image_path = os.path.join(image_dir, f"patch_{point_id}.jpeg")

    lon, lat = points_df[points_df['point_id'] == point_id][['longitude', 'latitude']].values[0]
    tile = mercantile.tile(lon, lat, 16)
    bbox = mercantile.bounds(tile)

    density, sparsity, mask = compute_density_from_vector_file(vector_path, bbox)
    print(f"Point {point_id}: Density = {density:.4f}, Sparsity = {sparsity:.4f}")

    osm_img = Image.open(image_path)

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(osm_img)
    axs[0].set_title("OSM Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("OSM Tag Raster Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


###

TILE_SIZE = 512  # Adjust according to your tile size

def compute_density_from_vector_file_for_tag(json_path, tile_bbox, target_tag):
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = []
    
    # Loop over all objects in the json file
    for obj in data.values():
        geo = obj[0]
        tags = obj[1]  # The list of tags associated with the geometry
        
        # Check if the target_tag is in the tags for the current geometry
        if target_tag in tags:
            geom = shape(geo)
            if not geom.is_valid:
                continue
            shapes.append((geom, 1))  # 1 = value for mask if the tag matches

    if not shapes:
        return 0.0, 1.0, np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    # Create transform from bounding box
    transform = Affine.translation(tile_bbox.west, tile_bbox.north) * Affine.scale(
        (tile_bbox.east - tile_bbox.west) / TILE_SIZE,
        -(tile_bbox.north - tile_bbox.south) / TILE_SIZE  # NOTE: negative scale for Y
    )

    # Rasterize the geometries onto the mask
    mask = rasterize(
        [(mapping(geom), value) for geom, value in shapes],
        out_shape=(TILE_SIZE, TILE_SIZE),
        transform=transform,
        fill=0,
        all_touched=True  # rasterize full coverage
    )

    # Calculate density and sparsity for the target tag
    density = mask.sum() / (TILE_SIZE * TILE_SIZE)
    sparsity = 1 - density

    return density, sparsity, mask

def compute_all_tags_density(json_path, tile_bbox):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle top-level wrapper
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
        data = next(iter(data.values()))

    shapes = []

    for obj in data.values():
        if not isinstance(obj, list) or len(obj) < 2:
            continue

        geo = obj[0]
        try:
            geom = shape(geo)
        except Exception:
            continue

        if not geom.is_valid:
            continue

        shapes.append(mapping(geom))  # just the geometry

    if not shapes:
        return 0.0, np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)

    # Affine transform from tile bounds to pixel grid
    transform = Affine.translation(tile_bbox.west, tile_bbox.north) * Affine.scale(
        (tile_bbox.east - tile_bbox.west) / TILE_SIZE,
        -(tile_bbox.north - tile_bbox.south) / TILE_SIZE
    )

    # Rasterize all geometries one by one and add up their presence
    tag_count_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
    for geom in shapes:
        single_mask = rasterize(
            [(geom, 1)],
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=transform,
            fill=0,
            all_touched=True
        )
        tag_count_mask += single_mask.astype(np.uint16)

    # Density: average tag coverage per pixel
    total_tags = tag_count_mask.sum()
    average_tags_per_pixel = total_tags / (TILE_SIZE * TILE_SIZE)

    return average_tags_per_pixel, tag_count_mask

def plot_and_save_tag_density(point_id, POINTS_DF, json_folder,  output_folder=None):
    json_path = json_folder+f"/{point_id}.json"
    lon, lat = POINTS_DF.loc[POINTS_DF['point_id'] == point_id, ['longitude', 'latitude']].values[0]
    tile_bbox = mercantile.bounds(mercantile.tile(lon, lat, 16))  # Example bounding box

    avg_density, tag_count_map = compute_all_tags_density(json_path, tile_bbox)

    plt.imshow(tag_count_map, cmap="plasma")
    plt.title(f"Avg tags per pixel: {avg_density:.3f}")
    plt.colorbar(label="Tag count per pixel")
    plt.axis("off")

    # Save
    if output_folder:
        output_path = output_folder + f"{point_id}_tagdensity.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

def plot_and_save_osm_with_mask(target_tag, point_id, POINTS_DF,  json_folder, image_folder,  output_folder=None):

    json_path = json_folder+f"/{point_id}.json"
    osm_image_path = image_folder + f"/patch_{point_id}.jpeg"

    # Extract the bounding box using the point_id from the dataframe
    lon, lat = POINTS_DF.loc[POINTS_DF['point_id'] == point_id, ['longitude', 'latitude']].values[0]
    tile_bbox = mercantile.bounds(mercantile.tile(lon, lat, 16))  # Example bounding box

    # Calculate density, sparsity, and mask
    density, sparsity, mask = compute_density_from_vector_file_for_tag(json_path, tile_bbox, target_tag)
    print(f"Tag: {target_tag}, Density: {density:.4f}, Sparsity: {sparsity:.4f}")

    # Load the corresponding OSM image for the tile
    osm_image = Image.open(osm_image_path)

    # Ensure the image is in RGB format and converted to a NumPy array
    osm_image = osm_image.convert("RGB")  # Convert to RGB if it's in RGBA or other formats
    osm_image = np.array(osm_image)

    # Check if the image is in the right format and dtype
    if osm_image.dtype != np.uint8:
        osm_image = osm_image.astype(np.uint8)

    # Plot the OSM image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(osm_image)

    # Create semi-transparent colored mask overlay (e.g., red)
    color_mask = np.zeros_like(osm_image, dtype=np.uint8)
    color_mask[..., 0] = 255  # Red
    alpha = np.where(mask > 0.5, 0.4, 0)  # Light transparency where mask is present
    ax.imshow(color_mask, alpha=alpha)

    # Draw outline
    ax.contour(mask, levels=[0.5], colors='red', linewidths=2)

    plt.title(f"Mask Outline for tag: {target_tag}")
    plt.axis("off")

    # Save
    if output_folder:
        output_path = output_folder + f"{target_tag}_{point_id}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def generate_images_for_tags(target_tags, POINTS_DF, json_folder, image_folder, output_root_folder, max_images_per_tag=10):
    # Loop through each tag
    for target_tag in target_tags:
        print(f"Processing tag: {target_tag}")
        
        # Create a folder for this tag if it doesn't exist
        tag_folder = os.path.join(output_root_folder, target_tag)
        os.makedirs(tag_folder, exist_ok=True)

        # Loop through the point_ids and check if they have the desired tag in their json file
        point_ids = POINTS_DF['point_id'].tolist()

        # Loop through the point_ids and generate images for those that have the tag
        images_generated = 0
        for point_id in point_ids:
            # Load the JSON file to find the tags for this point
            json_path = json_folder + f"/{point_id}.json"
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Loop over all objects in the json file to extract tags
                for obj in data.values():
                    geo = obj[0]
                    tags = obj[1]  # The list of tags associated with the geometry
                    
                    if target_tag in tags:
                        # If the tag matches, generate and save the image for this point
                        plot_and_save_osm_with_mask(target_tag, point_id, POINTS_DF, json_folder, image_folder, output_folder=f'{tag_folder}/')
                        images_generated += 1
                        break  # Once we find the tag, no need to check further objects in the JSON

            except FileNotFoundError:
                print(f"Warning: {json_path} not found.")
            
            # Stop after generating `max_images_per_tag` images
            if images_generated >= max_images_per_tag:
                break

def plot_all_masks_for_tile(point_id, json_path, image_path, POINTS_DF, max_tags=10, output_path=None):
    # Get bounding box
    lon, lat = POINTS_DF.loc[POINTS_DF['point_id'] == point_id, ['longitude', 'latitude']].values[0]
    tile_bbox = mercantile.bounds(mercantile.tile(lon, lat, 16))

    # Load base image
    osm_image = Image.open(image_path).convert("RGB")
    osm_image = np.array(osm_image).astype(np.uint8)

    # Load json and extract unique tags
    with open(json_path, "r") as f:
        data = json.load(f)

    tag_set = set()
    for obj in data.values():
        tags = obj[1]
        tag_set.update(tags)

    tags_to_plot = sorted(tag_set)[:max_tags]  # Cap number of tags

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(osm_image)

    # Generate distinct colors
    colormap = cm.get_cmap('tab10', len(tags_to_plot))
    tag_colors = {tag: colormap(i) for i, tag in enumerate(tags_to_plot)}
    legend_handles = []

    for tag in tags_to_plot:
        density, sparsity, mask = compute_density_from_vector_file_for_tag(json_path, tile_bbox, tag)
        if np.any(mask > 0.5):  # Only plot non-empty masks
            color = tag_colors[tag]
            color_rgb = np.array(color[:3]) * 255
            overlay = np.zeros_like(osm_image, dtype=np.uint8)
            overlay[..., :] = color_rgb
            alpha = np.where(mask > 0.5, 0.25, 0)
            ax.imshow(overlay, alpha=alpha)
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=1)
            legend_handles.append(mpatches.Patch(color=color, label=tag))

    # Add legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower right', fontsize=8)
    plt.title(f"Tag overlays for point {point_id}")
    plt.axis("off")

    # Save
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def convert_json_to_geojson(point_id, json_path, output_path):

    # Load input JSON
    with open(json_path, 'r') as f:
        input_json = json.load(f)

    # Initialize GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    # Convert the JSON data to GeoJSON format
    for key, value in input_json.get("0", {}).items():
        tags = value[1]

        # Skip features with unwanted tag
        if "boundary_administrative" in tags:
            continue

        geometry = value[0]
        properties = {"id": key}

        for tag in tags:
            properties[tag] = True

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": properties
        }
        geojson["features"].append(feature)

    # Save to output file
    output_file = os.path.join(output_path, f'{point_id}.geojson')
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"GeoJSON saved to: {output_file}")
