import os
import json


# Get the path to the upper folder
upper_folder_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Define the file path to the config.json in the upper folder
config_file_path = os.path.join(upper_folder_path, "config.json")

# Open and load the config.json file
with open(config_file_path) as f:
    config = json.load(f)
    folder_name = config['mp_results_path']

# Iterate over the files in the folder
for filename in os.listdir(folder_name):
    if filename.endswith('.png') and filename.startswith('shoulder'):
        # Remove the first two characters and the last characters from the file name
        new_filename = filename[2:-17] + '.png'
        
        # Rename the file
        os.rename(os.path.join(folder_name, filename), os.path.join(folder_name, new_filename))