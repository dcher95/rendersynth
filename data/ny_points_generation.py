import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def load_and_filter_counties(counties_file, nyc_counties):
    """
    Load the counties shapefile and filter it for the specified NYC counties.
    """
    counties_gdf = gpd.read_file(counties_file)
    counties_gdf = counties_gdf[counties_gdf['NAME'].isin(nyc_counties)]
    counties_gdf = counties_gdf.to_crs(epsg=2263)  # Reproject to NY Long Island feet
    return counties_gdf

def generate_grid_points(counties_gdf, spacing=1500):
    """
    Generate a grid of points within the bounding box of the NYC counties.
    Spacing defines the distance between points in meters (default: 600m).
    """
    minx, miny, maxx, maxy = counties_gdf.total_bounds
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)
    grid_points = [Point(x, y) for x in x_coords for y in y_coords]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=counties_gdf.crs)
    return grid_gdf

def filter_points_within_counties(grid_gdf, counties_gdf):
    """
    Filter out points that are not within the boundaries of the specified counties.
    """
    union_geom = counties_gdf.geometry.union_all()
    points_within = grid_gdf[grid_gdf.within(union_geom)].copy()
    return points_within


def assign_county_and_state(points_within, counties_gdf):
    """
    Assign the county and state ('New York') for each point in the GeoDataFrame.
    """
    points_within["county"] = points_within.apply(lambda row: counties_gdf[counties_gdf.contains(row.geometry)]['NAME'].values[0], axis=1)
    points_within["state"] = "New York"
    return points_within

def split_train_val(points_within, test_size=0.2):
    """
    Split the points into train and validation sets and return them as separate GeoDataFrames.
    """
    train_gdf, val_gdf = train_test_split(points_within, test_size=test_size, random_state=42, shuffle=True)
    return train_gdf, val_gdf

def save_csv(gdf, output_file):
    """
    Save the GeoDataFrame to a CSV file.
    """
    gdf[['point_id', 'latitude', 'longitude', 'county', 'state']].to_csv(output_file, index=False)

def plot_and_save(counties_gdf, train_gdf, val_gdf, output_image):
    """
    Plot the counties and the split points (train and val) and save the plot as an image file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    counties_gdf.to_crs(epsg=4326).plot(ax=ax, facecolor='none', edgecolor='black')
    train_gdf.plot(ax=ax, color='blue', markersize=3, label='Train')
    val_gdf.plot(ax=ax, color='red', markersize=3, label='Val')
    plt.legend()
    plt.title("NYC Sampled Points Split into Train/Val with County and State Info")
    plt.savefig(output_image, dpi=300)  # Save plot to file
    plt.close()

def main():
    # File paths
    counties_file = './data/Counties_Shoreline.zip'
    train_csv = 'train_points.csv'
    val_csv = 'val_points.csv'
    output_image = 'nyc_sampled_points_plot.png'

    # NYC counties to filter
    nyc_counties = ['Bronx', 'Kings', 'New York', 'Queens', 'Richmond']

    # Load and filter counties
    counties_gdf = load_and_filter_counties(counties_file, nyc_counties)

    # Generate grid points within counties
    grid_gdf = generate_grid_points(counties_gdf)

    # Filter points within counties
    points_within = filter_points_within_counties(grid_gdf, counties_gdf)

    # Assign county and state to each point
    points_within = assign_county_and_state(points_within, counties_gdf)

    # Get centroids and add lat/lon
    points_within["centroid"] = points_within.geometry
    points_within = points_within.to_crs(epsg=4326)
    points_within["longitude"] = points_within.geometry.x
    points_within["latitude"] = points_within.geometry.y

    # Assign unique point_id to all points before splitting
    points_within = points_within.reset_index(drop=True)
    points_within["point_id"] = points_within.index

    # Split into train/val sets
    train_gdf, val_gdf = split_train_val(points_within)

    # Save train and validation CSVs
    save_csv(train_gdf, train_csv)
    save_csv(val_gdf, val_csv)

    # Plot and save the image
    plot_and_save(counties_gdf, train_gdf, val_gdf, output_image)

    print(f"Train and Val CSV files saved: {train_csv}, {val_csv}")
    print(f"Plot saved as: {output_image}")

if __name__ == "__main__":
    main()
