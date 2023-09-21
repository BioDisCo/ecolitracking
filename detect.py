import os
from pathlib import Path
import cv2
import logging
from conversions import yolo_to_opencv
import numpy as np

# Configure the logging module
logging.basicConfig(
    level=logging.ERROR,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format
)

# Create a logger
logger = logging.getLogger('logger')

# Template is initialized to be empty
template = None


def create_template(labels_path, config):
    logging.debug("Starting to create template ...")

    # TODO potentially change this to supervision
    # List that stores the bounding boxes
    first_frame = {
        "frame_nr": 0,
        "cells": []
    }

    # load the first image to obtain the dimensions
    first_image_path = os.path.join(config['images_directory_path'], 'image000006.jpg')

    # Load an image from a file
    first_image = cv2.imread(first_image_path)

    # Check if the image was loaded successfully
    if first_image is None:
        logging.error("Could not load the first frame")
    else:
        # Get the height and width of the image
        image_height, image_width, _ = first_image.shape

        # convert to grayscale images
        gray_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)

        # Create a black mask with the same dimensions
        mask = np.zeros_like(gray_image, dtype=np.uint8)    
    
    # load the labels from the first frame
    first_image_labels_path = Path(os.path.join(labels_path, 'image000006.txt'))

    with open(first_image_labels_path, 'r') as file:
        # Read the contents of the file
        lines = file.readlines()

        # Start to fill in the cells for the first frame
        cells = first_frame['cells']
    
        for idx, line in enumerate(lines):
            line = line.strip()
            _, x, y, w, h = str.split(line, ' ')
            x, y, w, h = yolo_to_opencv(x, y, w, h, image_height, image_width)
            cells.append(
                {
                    "id": idx,
                    "parent": None,
                    "box": (x, y, w, h)
                }
            )

            logging.debug("Creating masks for inpainting ...")

            # Create a ROI mask where pixels are white if they meet the threshold condition
            temp_mask = (gray_image[y:y + h, x:x + w] < config['first_image_threshold']).astype(np.uint8) * 255

            # Update the original mask with the ROI mask
            mask[y:y + h, x:x + w] = temp_mask
    
    # Inpaint the cells in the first frame
    template = cv2.inpaint(first_image, mask, 20, cv2.INPAINT_NS)

    for cell in cells:
        x, y, w, h = cell['box']
        
        # Extract the the environment of the bounding box
        roi = template[y:y+h+config['region_increase_for_blurring'], x:x+w+config['region_increase_for_blurring']]

        # Blur the inoainted regions twice
        for _ in range(2):
            blurred_roi = cv2.GaussianBlur(roi, (3, 3), 30)  # Adjust the kernel size as needed
            roi = blurred_roi

        # Replace the ROI in the original image with the blurred ROI
        template[y:y+h+5, x:x+w+5] = blurred_roi

    # Save template
    application_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(application_directory, 'template.jpg')
    cv2.imwrite(template_path, template)

    # TODO annotate first frame as well

    return first_frame, template



            

def detect(image, template, config):
    boxes = []

    subtraction = cv2.subtract(template, image)
    subtraction = cv2.cvtColor(subtraction, cv2.COLOR_BGR2GRAY)


    _, ecoli_noisy = cv2.threshold(subtraction, config['subtraction_image_threshhold'], 255, cv2.THRESH_BINARY)

    # Apply median blur to the image
    ecoli = cv2.medianBlur(ecoli_noisy, 3)

    contours = cv2.findContours(ecoli, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        if h <= config['height_cutoff'] or w <= config['width_cutoff']:
            continue

        # TODO potentially add morph. opening if bounding box is large to better detect cell dup.


        # Increase the bounding box to make up for subtraction
        x -= 1
        y -= 1
        w += 2
        h += 2

        boxes.append((x, y, w, h))
    
    return boxes