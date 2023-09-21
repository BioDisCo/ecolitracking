from track import init
from detect import create_template, detect
from pathlib import Path
from conversions import opencv_to_yolo
import os
import cv2

if __name__ == "__main__":
    labels_path, config = init()
    _, template = create_template(labels_path, config)


    # Enumerate over all files in the directory
    for idx, image_file in enumerate(os.listdir(config['images_directory_path'])):
        # Skip the first frame since it was labeled manually
        if idx == 0:
            continue

        image_path = os.path.join(config['images_directory_path'], image_file)
        
        # Load an image
        image = cv2.imread(image_path)

        # Get dimensions of image
        image_height, image_width, _ = image.shape

        # Construct path for the corresponding label file
        application_directory = Path(os.path.dirname(os.path.abspath(__file__)))
        labels_path = Path(os.path.join(application_directory, 'labels'))
        label_file_path = Path(os.path.join(labels_path, image_file.split('.')[0] + '.txt'))

        # Obtain bounding boxes
        detected_boxes = detect(image, template, config)

        # Save labels
        with open(label_file_path, 'w') as label_file:
            for box in detected_boxes:
                x,y,w,h = box
                x,y,w,h = opencv_to_yolo(x,y,w,h, image_height, image_width)
                if label_file.tell() == 0:
                     label_file.write(f'0 {x} {y} {w} {h}')
                else: 
                    label_file.write(f'\n0 {x} {y} {w} {h}') 



