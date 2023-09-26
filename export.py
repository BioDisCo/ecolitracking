import logging
import os
import cv2
from pathlib import Path
import json
from moviepy.editor import ImageSequenceClip

# Configure the logging module
logging.basicConfig(
    level=logging.ERROR,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format
)

# Create a logger
logger = logging.getLogger('logger')




def create_video_from_images(image_folder, video_name, fps=32):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort images to maintain order

    # Create a list of image paths
    image_paths = [os.path.join(image_folder, img) for img in images]

    # Load images and create a video
    video = ImageSequenceClip(image_paths, fps=fps)
    video.write_videofile(video_name, fps=fps)

if __name__ == "__main__":

    application_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = Path(os.path.join(application_directory, 'config.json'))

    try:
        with open(config_path, 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        # Handle the case where the file does not exist
        logger.error(f"config.json is missing")

    image_folder = 'annotated_images' 
    video_name = config['output_name'] 

    create_video_from_images(image_folder, video_name)