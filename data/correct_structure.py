import json
import os
import logging


def convert_json_format(input_json_path, output_json_path, json_filename="0.json"):
    # Load the original JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Initialize an empty dictionary to hold the converted structure
    converted_data = {}

    # Iterate over each key-value pair in the original JSON
    for key, value in data.items():
        # Check if the key starts with 'n', 'w', 'r', or 'a'
        if key[0] in ['n', 'w', 'r', 'a']:
            # Determine the type from the prefix
            if key[0] == 'n':
                prefix_type = 'node'
            elif key[0] == 'w':
                prefix_type = 'way'
            elif key[0] == 'r':
                prefix_type = 'relation'
            elif key[0] == 'a':
                prefix_type = 'area'

            # Create the tuple for the converted key
            new_key = f"('{prefix_type}', {key[1:]})"  # Tuple in string form

            # If the file doesn't exist in the converted_data, create it
            key = json_filename.split('.')[0]
            if key not in converted_data:
                converted_data[key] = {}

            # Assign the value to the new structure under the correct file name
            converted_data[key][new_key] = value

    # Save the converted data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

    return converted_data  # Return the converted data for comparison


def process_files(input_folder, output_folder):
    # Get list of all JSON files in the input folder
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    # Process each file
    for input_filename in input_files:
        input_json_path = os.path.join(input_folder, input_filename)
        output_json_path = os.path.join(output_folder, input_filename)

        try:
            # Convert the input JSON and get the converted data
            converted_data = convert_json_format(input_json_path, output_json_path, json_filename=input_filename)
            
            # Check the number of keys in the input and output files
            with open(input_json_path, 'r') as f:
                input_data = json.load(f)
            
            # Compare the number of keys in both files
            if len(input_data) != len(converted_data[input_filename.split('.')[0]]):
                logging.error(f"Key count mismatch for {input_filename}: Input has {len(input_data)} keys, output has {len(converted_data)} keys.")
                print(f"Error: Key count mismatch for {input_filename}. Check the log for details.")
        
        except Exception as e:
            logging.error(f"Error processing {input_filename}: {e}")
            print(f"Error processing {input_filename}. Check the log for details.")

# Input and output folders
logging.basicConfig(filename='/data/cher/rendersynth/data/val/log.txt', level=logging.ERROR)
input_folder = '/data/cher/rendersynth/data/val/vector'
output_folder = '/data/cher/rendersynth/data/val/fixed_vector/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all files
process_files(input_folder, output_folder)
