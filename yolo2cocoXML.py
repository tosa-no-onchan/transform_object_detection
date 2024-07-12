"""
https://www.kaggle.com/code/siddharthkumarsah/convert-yolo-annotations-to-coco-pascal-voc


"""
# This code converts a YOLO-based dataset to the Pascal VOC format. 
# The script loops through each image file in the input directory 
# and loads the image using the PIL library. It then creates an XML
# annotation file for each image in the output directory. The annotations 
# are read from the corresponding YOLO format annotation file, and the 
# bounding box coordinates are converted to Pascal VOC format.

# The XML file contains the filename, image size, and bounding box
# coordinates for each object detected in the image. In this script,
# there is only one object class "bottle" with category ID 0.

import os
from PIL import Image

# Set the paths for the input and output directories
#input_dir = '/path/to/yolo/dataset'
input_dir = '/home/nishi/Documents/VisualStudio-TF/annotation/data-backup/雑草'
input_ano_dir = '/home/nishi/Documents/VisualStudio-TF/annotation/annotation/雑草'

#output_dir = '/path/to/voc/dataset'
output_dir = 'coco/dataset'

# Define the categories for the Pascal VOC dataset
categories = {"zasou": 0}

# Loop through the images in the input directory
for image_file in os.listdir(input_dir):

    # Load the image and get its dimensions
    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)
    width, height = image.size

    # Create the XML annotation file for the image
    annotation_file = open(os.path.join(output_dir, f"{image_file.split('.')[0]}.xml"), "w")
    annotation_file.write(f'\
        <annotation>\
            <filename>{image_file}</filename>\
            <size>\
                <width>{width}</width>\
                <height>{height}</height>\
                <depth>3</depth>\
            </size>\
            <object>\
                <name>zasou</name>\
                <bndbox>\
    ')

    # Load the bounding box annotations for the image
    #with open(os.path.join(input_dir, f'{image_file.split(".")[0]}.txt')) as f:
    with open(os.path.join(input_ano_dir, f'{image_file.split(".")[0]}.txt')) as f:
        annotations = f.readlines()

    # Loop through the annotations and add them to the XML file
    for ann in annotations:
        x, y, w, h = map(float, ann.strip().split()[1:])
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
        annotation_file.write(f'\
            <xmin>{x_min}</xmin>\
            <ymin>{y_min}</ymin>\
            <xmax>{x_max}</xmax>\
            <ymax>{y_max}</ymax>\
        ')

    # Close the XML annotation file
    annotation_file.write(f'\
                </bndbox>\
            </object>\
        </annotation>\
    ')
    annotation_file.close()

