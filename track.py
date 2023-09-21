import logging
import json
import os
import subprocess
import numpy as np
from pathlib import Path
from detect import create_template, detect
from measures import intersection_over_union, intersection_over_area
import cv2
import pandas as pd
import supervision as sv
from supervision.draw.color import Color 
from forest import init_forest, enter_cell, duplicate_cell, exit_cell, stop_forest, prune_forest, save_forest
from conversions import frame_to_time


# Configure the logging module
logging.basicConfig(
    level=logging.ERROR,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define the log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the date and time format
)

# Create a logger
logger = logging.getLogger('logger')


def init():
    logger.debug(f'Loading config ...')

    application_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = Path(os.path.join(application_directory, 'config.json'))

    try:
        with open(config_path, 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        # Handle the case where the file does not exist
        logger.error(f"config.json is missing")

    logger.info(f"First you need to label the first frame")
    logger.info(f"This application will create a folder named labels in the same directory your image folder is located in")
    logger.info(f"Once labelImg has opened press the 'Change Save Dir' button and select the labels folder that has just been created")
    logger.info(f"You can now start to label the image. Once you are finished press 'save' and close labelImg")


    first_image_path = os.path.join(config['images_directory_path'], 'image000000.jpg')

    label_command = f'labelImg {first_image_path}'

    # Create labels path in case it doesn't exist
    labels_path = Path(os.path.join(application_directory, 'labels'))

    if not labels_path.exists():
        labels_path.mkdir(parents=True, exist_ok=True)

    # Create classes file in case it doesn'exist
    classes_path = Path(os.path.join(labels_path, 'classes.txt'))

    if not labels_path.exists():
        try:
            # Try to open the file in write mode (creates it if it doesn't exist)
            with open(classes_path, 'w') as file:
                pass

        except IOError as e:
            logger.error("classes file couldnt be created")

    # Open labelImg
    subprocess.run(label_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Create dict for storing tree data
    tree_data = []

    tree_data_path = Path(os.path.join(application_directory, 'tree_data.json'))

    logging.debug("init finished")

    return labels_path, config, tree_data

def draw_detected_boxes(image, detected_boxes, previous_labels):
    old_boxes = [cell["box"] for cell in previous_labels['cells']]
    draw_image = image.copy()

    # for old_box in old_boxes:
    #     x,y,w,h = old_box
    #     cv2.rectangle(draw_image, (x, y), (x + w, y + h), (0, 0, 255), 1) 
    
    # for new_box in detected_boxes:
    #     x,y,w,h = new_box
    #     new_box = cv2.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # cv2.imshow('Old and new boxes', draw_image)
    # cv2.waitKey(0)

def get_merges(mapping):

    # Find values in 't+1' column that appear more than once
    duplicates = mapping[mapping.duplicated(subset=['t+1'], keep=False)]

    # Create a dictionary to store the values and their corresponding 't' entries
    result_dict = {}

    # Iterate through the DataFrame and populate the dictionary
    for _, row in duplicates.iterrows():
        t_plus_1_value = row['t+1']
        t_value = row['t']

        if t_plus_1_value not in result_dict:
            result_dict[t_plus_1_value] = [t_value]
        else:
            result_dict[t_plus_1_value].append(t_value)

    return  result_dict



def check_border(box, image_height, image_width, config):
    x, y, w, h = box
    distance = min(
        # Distance to left boundary
        x,
        
        # Distance to right boundary
        image_width - (x + w),

        # Distance to lower boundary
        image_height - (y + h),
        
        # Distance to upper boundary
        y,
    )

    return distance < config["border_distance_cutoff"]


def get_cell_by_id(cell_id, labels):
    return next(filter(lambda cell: cell['id'] == cell_id, labels['cells']))


def sliding_window_search(previous_image, box, current_image, config, image_height, image_width, merged_box = None):
    x, y, w, h = box

    # variable to keep track of the best ssd score
    best_ssd = float('inf')

    # variable to keep track of the best candidate window
    best_window = None

    # Extract the target window inside the box from previous_image
    target_window = previous_image[y:y+h, x:x+w]
    
    # Get the region we serach in
    if merged_box is not None:
        # merged box
        search_x, search_y, search_w, search_h = merged_box
    else:
         # If we search for bacteria that disappeared we search in the local environment of its old position
        search_w = w + config['sliding_window_search_width_increase']
        search_h = h + config['sliding_window_search_height_increase']
        search_x = x - config['sliding_window_search_width_increase'] // 2
        search_y = y - config['sliding_window_search_height_increase'] // 2
    
    # Iterate over possible positions for the candidate window
    # We use min and max just to make sure we don't slide out of bounds
    for candidate_y in range(max(0, search_y), min(image_height, search_y + search_h - h + 1)):
        for candidate_x in range(max(0, search_x), min(image_width, search_x + search_w - w + 1)):
            # Extract the candidate window from current_image
            candidate_window = current_image[candidate_y:candidate_y+h, candidate_x:candidate_x+w]
            # Compute the SSD between target_window and candidate_window
            ssd = np.sum(np.square(target_window - candidate_window))

            if ssd < best_ssd:
                best_ssd = ssd
                best_window = candidate_x, candidate_y, w, h
    xb, yb, wb, hb = best_window
    test_image = current_image.copy()
    old_image = previous_image.copy()
    cv2.rectangle(old_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.rectangle(test_image, (search_x, search_y), (search_x + search_w, search_y + search_h), (0, 255, 0), 1)
    cv2.rectangle(test_image, (xb, yb), (xb + wb, yb + hb), (0, 0, 255), 1)
    # cv2.imshow('Best Window', test_image)
    # cv2.imshow('Search for', old_image)
    # cv2.waitKey(0)
    return best_window
 

def annotate_frame(current_labels, image, image_height, image_width, image_file):
        application_directory = Path(os.path.dirname(os.path.abspath(__file__)))
        target_dir_path = Path(os.path.join(application_directory, 'annotated_images'))

        xyxy = np.array([cell["box"] for cell in current_labels['cells']])

        # Convert coordinates to (x1,y1,x2,y2) format
        xyxy[:,2] += xyxy[:,0] 
        xyxy[:,3] += xyxy[:,1]

        class_id = np.zeros(xyxy.shape[0], dtype=int)
        
        tracker_id = np.array([cell["id"] for cell in current_labels['cells']])

        detections = sv.Detections(xyxy=xyxy, class_id=class_id, tracker_id=tracker_id)

        box_annotator = sv.BoxAnnotator(
            thickness=1,
            text_scale=0.25,
            text_thickness=0,
            text_padding=1,
            text_color=Color.white()
        )
        labels = [
	        f"Ecoli {tracker_id}"
	        for _, _, _, _, tracker_id
	        in detections
        ]

        annotated_frame = box_annotator.annotate(
	        scene=image.copy(),
	        detections=detections,
	        labels=labels
        )
        with sv.ImageSink(target_dir_path=target_dir_path, overwrite=False) as sink:
            sink.save_image(image=annotated_frame, image_name=image_file)
        # cv2.imshow('annotated image', annotated_frame)

def track(first_frame_labels, config, template):
    previous_labels = first_frame_labels
    

    # Enumerate over all files in the directory
    for frame_nr, image_file in enumerate(os.listdir(config['images_directory_path'])):
        # Skip the first frame since it was labeled manually
        if frame_nr == 0:
            # Previous image is needed for sliding window search
            previous_image_file = os.listdir(config['images_directory_path'])[frame_nr]
            previous_image = cv2.imread(os.path.join(config['images_directory_path'], previous_image_file))
            max_id = max(previous_labels['cells'], key=lambda x: x["id"])["id"]
            annotate_frame(first_frame_labels, previous_image, previous_image.shape[0], previous_image.shape[1], image_file)
            
            # Set up forest
            init_forest()
            in_frame = []
            time_offsets = []

            for cell in first_frame_labels['cell']:
                enter_cell(cell['id'], 0, time_offsets, in_frame)
            
            continue
        
        # cells with IDs are stored here
        current_labels = {
            "frame_nr": frame_nr,
            "cells": []
        }

        image_path = os.path.join(config['images_directory_path'], image_file)
        
        # Load an image
        image = cv2.imread(image_path)

        # Get dimensions of image
        image_height, image_width, _ = image.shape

        # Obtain bounding boxes
        detected_boxes = detect(image, template, config)

        draw_detected_boxes(image, detected_boxes, previous_labels)

        # List for the mapping of bounding boxes
        mapping = []

        # List for storing cells that duplicated
        duplicated = []
        
        # List for storing cells that merged
        merged = []



        for cell in previous_labels['cells']:
            box1 = cell['box']
            # Create an empty DataFrame with two columns
            scores = pd.DataFrame(columns=['IoU', 'IoA'])  
            for idx, box2 in enumerate(detected_boxes):
                IoU = intersection_over_union(box1, box2)
                IoA = intersection_over_area(box1, box2)
                scores.loc[idx] = [IoU, IoA]
            
            # Check if cell duplication took place
            scores = scores.sort_values(by='IoA', ascending=False)

            # Obtain the two boxes with the highest IoA score and check if duplication happened
            highest_IoA = scores['IoA'].head(2)
            
            # Obtain the indices of the two boxes
            cell_indices = highest_IoA.index.to_list()

            if (highest_IoA > config['IoA_threshold']).all():
                logger.info("duplication happened!")

                # Save cell_id for mapping loop
                duplicated.append(cell['id'])

                # Update forest
                duplicate_cell(cell['id'], frame_to_time(frame_nr), max_id, in_frame)

                mapping.append([cell['id'], cell_indices[0]])
                mapping.append([cell['id'], cell_indices[1]])
                
                max_id += 1

                current_labels['cells'].append(
                    {
                        "id": max_id,
                        "parent": cell['id'],
                        "box": detected_boxes[cell_indices[0]]
                    }
                )

                max_id += 1

                # The other cell is considered new
                current_labels['cells'].append(
                    {
                        "id": max_id,
                        "parent": cell['id'],
                        "box": detected_boxes[cell_indices[1]]
                    }
                ) 
                pass

            # Check if the greatest overlap is significant enough
            if scores['IoU'].max() < config['IoU_threshold']:
                # target_index = scores['IoU'].idxmax()
                # test_image = image.copy()
                # old_x,old_y,old_w,old_h = cell['box']
                # new_x,new_y,new_w,new_h = detected_boxes[target_index]
                # cv2.rectangle(test_image, (old_x, old_y), (old_x + old_w, old_y + old_h), (0, 0, 255), 1)
                # cv2.rectangle(test_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 1)
                # cv2.imshow('No Mapping', test_image)
                # cv2.waitKey(0)
                continue 

            target_index = scores['IoU'].idxmax()                                                                                                                                                                                                                                                                                                                                                         
            cell_id = cell['id']

            # test_image = image.copy()
            # old_x,old_y,old_w,old_h = cell['box']
            # new_x,new_y,new_w,new_h = detected_boxes[target_index]
            # cv2.rectangle(test_image, (old_x, old_y), (old_x + old_w, old_y + old_h), (0, 0, 255), 1)
            # cv2.rectangle(test_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 1)
            # cv2.imshow('Mapping', test_image)
            # cv2.waitKey(0)

            if not cell_id in duplicated:
                mapping.append([cell_id, target_index])
 
        mapping = pd.DataFrame(mapping)
        mapping.columns = ['t', 't+1']
        
        # Check if merging took place
        for merged_cell_index, preimages in get_merges(mapping).items():
            # new box from merging in which we search
            merged_box = detected_boxes[merged_cell_index]
            for cell_id in preimages:
                # Save for later so we can skip it in mapping loop
                merged.append(cell_id)

                # box for which we need to search
                box = get_cell_by_id(cell_id, previous_labels)['box']
                new_box = sliding_window_search(previous_image, box, image, config, image_height, image_width)

                current_labels['cells'].append(
                    {
                        "id": cell_id,
                        "parent": None,
                        "box": new_box
                    }
                ) 

        # Check if any new boxes appeared
        boxes_not_mapped_onto = [value for value in range(len(detected_boxes)) if value not in mapping["t+1"].unique()]

        for index in boxes_not_mapped_onto:
            # Get bounding box
            box = detected_boxes[index]
            # Box is only kept if its close enough to the border
            if check_border(box, image_height, image_width, config):
                # New cell gets new id
                max_id += 1

                # Update forest
                enter_cell(max_id, frame_to_time(frame_nr), time_offsets, in_frame)

                current_labels['cells'].append(
                    {
                        "id": max_id,
                        "parent": None,
                        "box": box
                    }
                ) 
            # Otherwise it's probably dirt or something 
        
        
        # Check if boxes disappeared
        cell_ids = [cell['id'] for cell in previous_labels['cells']]
        # Use list comprehension to filter 'ids' based on 't' column
        boxes_not_mapped = [cell_id for cell_id in cell_ids if cell_id not in mapping['t'].values]

        for cell_id in boxes_not_mapped:
            # Get bounding box
            box = get_cell_by_id(cell_id, previous_labels)['box']
            
            # If box was lost in the middle of the window we search it again
            if not check_border(box, image_height, image_width, config):
                new_box = sliding_window_search(previous_image, box, image, config, image_height, image_width)
                current_labels['cells'].append(
                    {
                        "id": cell_id,
                        "parent": None,
                        "box": new_box
                    }
                )

                continue 
            # If box was close to border it probably left the window

            # Update forest
            exit_cell(cell_id, frame_to_time(frame_nr), in_frame)


        for _, row in mapping.iterrows():
            cell_id = row['t'] 
            if cell_id in duplicated or cell_id in merged:
                continue

            current_labels['cells'].append(
                {
                    "id": cell_id,
                    "parent": None,
                    "box": detected_boxes[row['t+1']]
                }
            )

            enter_cell(cell_id, frame_to_time(frame_nr), time_offsets, in_frame) 


        # Annotate current frame
        annotate_frame(current_labels, image, image_height, image_width, image_file)

        # Update for next iteration
        previous_image = image.copy()
        previous_labels = current_labels.copy()
    
    stop_forest(in_frame, frame_to_time(frame_nr))
    prune_forest()
    save_forest()

                
        


if __name__ == "__main__":
    labels_path, config, tree_data = init()
    first_frame_labels, template = create_template(labels_path, config)
    track(first_frame_labels, config, template, tree_data)

    # image = cv2.imread('images/image000006.jpg')
    # detect(image, template, config)