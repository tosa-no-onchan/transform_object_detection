"""
https://www.kaggle.com/code/siddharthkumarsah/convert-yolo-annotations-to-coco-pascal-voc

# The COCO dataset is saved as a JSON file in the output directory.

"""
import json
import os
from PIL import Image

# Set the paths for the input and output directories
#input_dir = '/path/to/yolo/dataset'
input_dir = '/home/nishi/Documents/VisualStudio-TF/annotation/data-backup/雑草'
input_ano_dir = '/home/nishi/Documents/VisualStudio-TF/annotation/annotation/雑草'

#output_dir = '/path/to/coco/dataset'
output_dir = 'coco/dataset'

# Define the categories for the COCO dataset
#categories = [{"id": 0, "name": "bottle"}]
categories = [{"id": 0, "name": "zasou"}]

# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

# Loop through the images in the input directory
for image_file in os.listdir(input_dir):
    
    # Load the image and get its dimensions
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)
    width, height = image.size
    
    # Add the image to the COCO dataset
    image_dict = {
        "id": int(image_file.split('.')[0]),
        "width": width,
        "height": height,
        "file_name": image_file
    }
    coco_dataset["images"].append(image_dict)
    
    # Load the bounding box annotations for the image
    #with open(os.path.join(input_dir, f'{image_file.split(".")[0]}.txt')) as f:
    with open(os.path.join(input_ano_dir, f'{image_file.split(".")[0]}.txt')) as f:
        annotations = f.readlines()
    
    # Loop through the annotations and add them to the COCO dataset
    for ann in annotations:
        x, y, w, h = map(float, ann.strip().split()[1:])
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
        ann_dict = {
            "id": len(coco_dataset["annotations"]),
            "image_id": int(image_file.split('.')[0]),
            "category_id": 0,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        }
        coco_dataset["annotations"].append(ann_dict)

# Save the COCO dataset to a JSON file
with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
    json.dump(coco_dataset, f)
