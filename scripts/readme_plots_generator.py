import os
import json

def generate_readme():
    # Read the configuration from config.json
    with open('config.json') as f:
        config = json.load(f)
    
    # Get the folder path where the .png files are located
    folder_name = config['data_path']
    
   
    with open("README.md", "w") as readme_file:
        # Write main header for lift name
        readme_file.write("# " + lift_name + "\n\n")
        
        # Iterate over joint names
        for joint_name in joint_names:
            readme_file.write("## " + joint_name.capitalize() + "\n\n")
            
            # Get plots for the current joint name
            plots = get_plots(folder_name, joint_name)
            
            # Iterate over plots and write titles with image links
            for plot in plots:
                video_origin = get_video_origin(plot)
                
                # Write small title for the plot
                readme_file.write("### " + video_origin + "\n\n")
                
                # Add image link to the README
                image_link = get_image_link(folder_name, plot)
                readme_file.write("![" + video_origin + "](" + image_link + ")\n\n")

def get_plots(folder_name, joint_name):
    # Get all plot files in the folder
    plot_files = [file for file in os.listdir(folder_name) if file.endswith(".png")]
    
    # Filter plot files based on joint name
    plots = [file for file in plot_files if joint_name in file]
    
    return plots

def get_video_origin(plot):
    # Extract video origin from the plot filename
    video_origin = plot.split("_")[2]
    
    return video_origin

def get_image_link(folder_name, plot):
    # Generate relative path to the image file
    image_path = os.path.join(folder_name, plot)
    
    # Convert path to Unix-style for Markdown compatibility
    image_link = image_path.replace("\\", "/")
    
    return image_link

# Generate the README.md file
generate_readme()