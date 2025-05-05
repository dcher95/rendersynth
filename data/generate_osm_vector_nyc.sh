#!/bin/bash

source config.env

# Define input file
INPUT_FILE=$OSM_PBF

# Create intermediate output directories
mkdir -p $FILTERED_PBF_DIR
mkdir -p $FILTERED_GEOJSON_DIR
mkdir -p $FILTERED_JOSN_DIR
mkdir -p $CLIPPED_GEOJSON_DIR
mkdir -p $BOUNDING_BOXES_DIR
mkdir -p $CLIPPED_BBOX_GEOJSON_DIR

# List of top-level OSM tags to extract
IFS=',' read -r -a TAG_ARRAY <<< "$TAGS"

# Path to NYC boundary file (must be in EPSG:4326)
BOUNDARY_FILE=$BOUNDARY_GEOJSON

# Generate the bounding boxes using the Python script
echo "Generating bounding boxes..."
python generate_bboxes.py

# List bounding boxes from the geojson file
BOUNDING_BOXES=$(jq -r '.features[].properties.point_id' ${COMBINED_BBOX_OUTPUT_PATH})

echo "Starting filtering for all OSM top-level tags..."

for TAG in "${TAG_ARRAY[@]}"
do
    echo "Processing tag: $TAG"

    PBF_OUT="${FILTERED_PBF_DIR}/${TAG}.osm.pbf"
    GEOJSON_OUT="${FILTERED_GEOJSON_DIR}/${TAG}.geojson"
    CLIPPED_NYC_OUT="${CLIPPED_GEOJSON_DIR}/${TAG}_clipped.geojson"

    # Step 1: Filter to .osm.pbf
    if [ -f "$PBF_OUT" ]; then
        echo "  Skipping PBF filter (exists: $PBF_OUT)"
    else
        echo "  Filtering PBF..."
        osmium tags-filter "$INPUT_FILE" "${TAG}=*" -o "$PBF_OUT" --overwrite
    fi

    # Step 2: Export to .geojson
    if [ -f "$GEOJSON_OUT" ]; then
        echo "  Skipping GeoJSON export (exists: $GEOJSON_OUT)"
    else
        echo "  Exporting to GeoJSON..."
        osmium export --add-unique-id=type_id "$PBF_OUT" -o "$GEOJSON_OUT" -f geojson --overwrite
    fi

    # Step 3: Clip to NYC boundary
    if [ -f "$CLIPPED_NYC_OUT" ]; then
        echo "  Skipping NYC clip (exists: $CLIPPED_NYC_OUT)"
    else
        echo "  Clipping to NYC boundary..."
        ogr2ogr -f GeoJSON "$CLIPPED_NYC_OUT" "$GEOJSON_OUT" -clipsrc "$BOUNDARY_FILE"
    fi

    # Step 4: Clip using each bounding box
    for BBOX_ID in $BOUNDING_BOXES; do
        BBOX_FILE="${BOUNDING_BOXES_DIR}/${BBOX_ID}.geojson"
        BBOX_OUT="${CLIPPED_BBOX_GEOJSON_DIR}/${BBOX_ID}/${TAG}.geojson"
        mkdir -p "${CLIPPED_BBOX_GEOJSON_DIR}/${BBOX_ID}"
        
        if [ -f "$BBOX_OUT" ]; then
            echo "    Skipping bbox clip for $BBOX_ID (exists: $BBOX_OUT)"
        else
            echo "    Clipping to bbox: $BBOX_ID"
            ogr2ogr -f GeoJSON "$BBOX_OUT" "$CLIPPED_NYC_OUT" -clipsrc "$BBOX_FILE"
        fi
    done

done


# Combine bounding box in specific way to create new JSON for each point using Python script
echo "Combining OSM tags for each bounding box..."
python combine_osm_tags.py

# Remove intermediate .osm.pbf files
echo "Cleaning up intermediate directories..."
rm -r ./data/filtered_pbf
# rm -r ./data/filtered_geojson
# rm -r ./data/clipped_geojson
# rm -r ./data/clipped_bbox_geojson
# rm -r ./data/bounding_boxes

echo "All tags processed and clipped. Output is in ./data/vector"
