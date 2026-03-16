"""
yolo2huggingface_metadata2.py

YOLO annotations are convert to COCO huggingface metadata and saved as a JSONL file in the output directory.

transform_object_detection/yolo2huggingface_metadata.py

https://www.kaggle.com/code/siddharthkumarsah/convert-yolo-annotations-to-coco-pascal-voc
https://huggingface.co/docs/datasets/main/en/image_dataset#loading-script

"""
import json
import os
from PIL import Image

# Set the paths for the input and output directories
#input_dir = '/path/to/yolo/dataset'
input_dir = '/home/nishi/Documents/VisualStudio-TF/annotation/data-backup'
input_ano_dir = '/home/nishi/Documents/VisualStudio-TF/annotation/annotation'

#output_dir = '/path/to/coco/dataset'
output_dir = 'datasets'


"""
datasets/
  train/
    zasou/
      xxxx.img
      xxxx.img
    tree/
      xxxx.img
      xxxx.img
    potted_plant/
      xxxx.img
      xxxx.img
    metadata.jsonl    <-- train の下
  test/            # もし、 test があれば
    zasou/
      xxxx.img
      xxxx.img
    tree/
      xxxx.img
      xxxx.img
    potted_plant/
      xxxx.img
      xxxx.img
    metadata.jsonl    <-- test の下
"""

"""
metadata.jsonl

{"img_dir":"雑草","file_name": "zasou/0001.png", "objects": {"bbox": [[302.0, 109.0, 73.0, 52.0]], "categories": [0]}}
{"img_dir":"樹木","file_name": "tree/0002.png", "objects": {"bbox": [[810.0, 100.0, 57.0, 28.0]], "categories": [1]}}
{"img_dir":"庭の鉢植えの植物","file_name": "potted_plant/0003.png", "objects": {"bbox": [[160.0, 31.0, 248.0, 616.0], [741.0, 68.0, 202.0, 401.0]], "categories": [2, 2]}}

from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="datasets")

print(dataset['train'][0])

{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=943x663 at 0x7F9EC9E77C10>,
 'objects': {
  'bbox': [[302.0, 109.0, 73.0, 52.0],
   [810.0, 100.0, 57.0, 28.0],
   [160.0, 31.0, 248.0, 616.0],
   [741.0, 68.0, 202.0, 401.0]],
  'category': [4, 4, 0, 0]}}

"""

img_dirs = ['雑草','樹木','庭の鉢植えの植物']
img_sub_paths = ['zasou','tree','potted_plant']

# Define the categories for the COCO dataset
categories = [{"id": 0, "name": "zasou"},
              {"id": 1, "name": "tree"},
              {"id": 2, "name": "potted_plant"}
              ]

metadata=[]

lc=0

objects_id=0

img_id=0

def conv(img_dir,img_sub_path,input_dir,input_ano_dir):
    global objects_id
    global lc,img_id
    # Loop through the images in the input directory
    for image_file in os.listdir(input_dir):
        
        # Load the image and get its dimensions
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size

        line={}

        #print('image_file:',image_file)
        #line['objects']={}
        
        # Load the bounding box annotations for the image
        image_id=image_file.split(".")[0]
        ano_txt=os.path.join(input_ano_dir, f'{image_file.split(".")[0]}.txt')
        if not os.path.exists(ano_txt):
            continue

        with open(os.path.join(input_ano_dir, f'{image_file.split(".")[0]}.txt')) as f:
            annotations = f.readlines()

        #print("annotations:",annotations)
        objects={}
        objects["id"]=[]
        objects["area"]=[]
        objects["bbox"]=[]
        objects["category"]=[]

        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            cls = int(ann.strip().split()[0])
            #print("cls:",cls,"x:",x," y:",y ," w:",w,"h:",h)
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)

            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            objects["area"].append(int((x_max-x_min)*(y_max-y_min)))
            objects["bbox"].append(bbox)
            objects["category"].append(cls)
            objects["id"].append(objects_id)
            objects_id+=1

        #line['image_id']=image_id
        line['img_dir']=img_dir
        line['img_sub_path']=img_sub_path
        line['file_name']=img_sub_path+'/'+image_file
        #line['image_id']=int(image_id)
        line['image_id']=img_id
        line['width']= width
        line['height']= height
        line['objects']=objects
        metadata.append(line)

        lc +=1
        img_id+=1
        #if lc > 4:
        #    break

for img_dir,img_sub_path in zip(img_dirs,img_sub_paths):
    input_dir_x = os.path.join(input_dir,img_dir)
    input_ano_dir_x = os.path.join(input_ano_dir,img_dir)
    conv(img_dir,img_sub_path,input_dir_x,input_ano_dir_x)

#for s in metadata[:5]:
#    print("s:", s)

if True:
    # Save the huggingface metadata to a JSONL file
    with open(os.path.join(output_dir,'train','metadata.jsonl'), 'w') as f:
        #f.writelines([json.dumps(l) for l in datasets])
        for l in metadata:
            f.writelines(json.dumps(l)+'\n')