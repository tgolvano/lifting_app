import os
import json

with open('config.json') as f:
    config = json.load(f)
for file_name in os.listdir(config['data_path']):
    # check if the file is a .mp4 video and its name doesn't start with 'bad_form_snatches'
    if file_name.endswith('.mp4') and not file_name.startswith('bad_form_snatches'):
        # create the full path to the video
        video_path = os.path.join(config['data_path'], file_name)
    print (f"{video_path}")