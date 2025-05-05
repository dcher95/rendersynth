import geopandas as gpd

COUNTIES_FILE = './data/Counties_Shoreline.zip'
NYC_COUNTIES = ['Bronx', 'Kings', 'New York', 'Queens', 'Richmond']
OUTPUT_BOUNDARY = './data/boundary_nyc.geojson'
OUTPUT_UNION = './data/boundary_nyc_union.geojson'

def load_and_filter_counties(counties_file, nyc_counties):
    """
    Load the counties shapefile and filter it for the specified NYC counties.
    """
    counties_gdf = gpd.read_file(counties_file)
    counties_gdf = counties_gdf[counties_gdf['NAME'].isin(nyc_counties)]
    counties_gdf = counties_gdf.to_crs(epsg=4326)  # ogr2ogr prefers EPSG:4326
    return counties_gdf

# Load and filter
counties_gdf = load_and_filter_counties(COUNTIES_FILE, NYC_COUNTIES)

# Save to GeoJSON
counties_gdf.to_file(OUTPUT_BOUNDARY, driver='GeoJSON')

nyc_union = counties_gdf.unary_union  # This is a single shapely geometry
nyc_union_gdf = gpd.GeoDataFrame(geometry=[nyc_union], crs="EPSG:4326")
nyc_union_gdf.to_file(OUTPUT_UNION, driver='GeoJSON')

