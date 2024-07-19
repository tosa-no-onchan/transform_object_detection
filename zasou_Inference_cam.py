# -*- coding: utf-8 -*-
"""

Visualstudio-torch_env/transform_object_detection/zasou_Inference_cam.py
"""
from transformers import pipeline
import requests
import numpy as np
import os
import sys
import time

import cv2

from PIL import Image, ImageDraw

#check_point="./detr-resnet-50_finetuned_zasou"
check_point="tosa-no-onchan/detr-resnet-50_finetuned_zasou"
#check_point="./detr-resnet-50_finetuned_zasou-300+100epoch"
obj_detector = pipeline("object-detection", model=check_point, device=0)



path='./datasets/train/zasou'

def one_shot():

    flist=os.listdir(path)
    cnt=0

    #url = "https://i.imgur.com/2lnWoly.jpg"
    #image = Image.open(requests.get(url, stream=True).raw)
    #image = Image.open('./datasets/train/zasou/11.jpeg')
    image = Image.open(path+'/'+flist[cnt])
    cnt+=1
    #image

    results = obj_detector(image)
    print(results)

    draw = ImageDraw.Draw(image)
    for i in range(len(results)):
        score = results[i]['score']
        label = results[i]['label']
        box=results[i]['box']
        (x,y,x2,y2) =box.values()
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), label, fill="white")

    image.show()

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def cam_movie():

  camera_id=0
  width=640
  height=480

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  thresh=0.7

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    counter += 1

    imagex=cv2pil(image)

    results = obj_detector(imagex)
    #print(results)

    for i in range(len(results)):
        score = results[i]['score']
        print('score:',score)
        if score < thresh:
           continue
        label = results[i]['label']
        box=results[i]['box']
        (x0,y0,x1,y1) =box.values()
        #draw.rectangle((x, y, x2, y2), outline="red", width=1)

        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0))

        #draw.text((x, y), label, fill="white")
        cv2.putText(image, label, (x0,y0), cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)


    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    detection_result_image=image
    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(detection_result_image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector',detection_result_image)

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    if True:
       cam_movie()
    else:
        one_shot()