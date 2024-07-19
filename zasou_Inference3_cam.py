# -*- coding: utf-8 -*-
"""

Visualstudio-torch_env/transform_object_detection/zasou_Inference3_cam.py

https://www.geeksforgeeks.org/object-detection-with-detection-transformer-dert-by-facebook/

https://github.com/huggingface/transformers/blob/main/docs/source/ja/model_doc/detr.md

"""

import torch
import torchvision.transforms as T
import requests
from PIL import Image, ImageDraw, ImageFont

from transformers import DetrForObjectDetection,AutoModelForObjectDetection
import os

import sys
import time

import cv2

from PIL import Image, ImageDraw
import numpy as np


# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#Normalize
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


check_point="./detr-resnet-50_finetuned_zasou-300+100epoch"

def one_shot():

  proc_type=0

  # https://github.com/huggingface/transformers/blob/main/docs/source/ja/model_doc/detr.md
  model = DetrForObjectDetection.from_pretrained(check_point)

  #print(type(model))
  #PATH="./detr-resnet-50_finetuned_zasou-300+100epoch/checkpoint-1600/rng_state.pth"

  #model = torch.load(PATH)
  model.eval()
  model = model.cuda()


  #print('type(model):',type(model))
  #print('model:',model)
  #print('model.bbox_predictor:',model.bbox_predictor)

  CLASSES = ['N/A', 'zasou']

  path='./datasets/train/zasou'
  flist=os.listdir(path)
  cnt=0

  #print(model)

  img = Image.open(path+'/'+flist[cnt])
  cnt+=1

  print('img.size:',img.size)
  x=img.size[::-1]
  print('x:',x)

  #target_sizes = torch.tensor([img.size[::-1]])
  #target_sizes = torch.tensor([[460,460]])

  img_tens = transform(img).unsqueeze(0).cuda()

  with torch.no_grad():
    output = model(img_tens)
    #output = model2(**inputs)

  if proc_type==0:
    # reffer from
    # /home/nishi/torch_env/lib/python3.10/site-packages/transformers/models/detr/image_processing_detr.py
    #   1503 def post_process(self, outputs, target_sizes):
    out_logits, out_bbox = output.logits, output.pred_boxes
    #print('out_logits:',out_logits)
    #print('out_logits.shape:',out_logits.shape)
    #print('out_bbox:',out_bbox)
    prob = torch.nn.functional.softmax(out_logits, -1)
    print('prob.shape:',prob.shape)
    #print('prob:',prob)

    #print('prob[..., :-1]:',prob[..., :-1])
    #print("prob[..., :-1].max(-1):",prob[..., :-1].max(-1))

    scores, labels = prob[..., :-1].max(-1)
    #print('scores:',scores)
    #print('labels:',labels)

    results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, out_bbox)]

    #print('results',results)

    draw = ImageDraw.Draw(img)

    idx = results[0]['scores'].argmax()

    acc = results[0]['scores'][idx]
    cls = results[0]['labels'][idx]
    box = results[0]['boxes'][idx]
    label = CLASSES[cls+1]
    print('label:',label)
    print('acc:',float(acc))
    x, y, w,h = box
    x = int(x * float(img.size[0]))
    w = int(w * float(img.size[0]))
    y = int(y * float(img.size[1]))
    h = int(h * float(img.size[1]))

    x0, x1 = x-w//2, x+w//2
    y0, y1 = y-h//2, y+h//2
    draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
    draw.text((x0, y0), label, fill='white')

    img.show()

    sys.exit()


    for acc,cls,box in zip(results[0]['scores'], results[0]['labels'],results[0]['boxes']):
      print('acc:',acc)
      if cls < 0:
        continue
      acc=float(acc)

      label = CLASSES[cls+1]
      print(label)
      #box = box.cpu() * torch.Tensor([800, 600, 800, 600])
      x, y, w,h = box
      x = int(x * float(img.size[0]))
      w = int(w * float(img.size[0]))
      y = int(y * float(img.size[1]))
      h = int(h * float(img.size[1]))

      x0, x1 = x-w//2, x+w//2
      y0, y1 = y-h//2, y+h//2
      draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
      draw.text((x0, y0), label, fill='white')

    img.show()

    sys.exit()



  #print('output[0]:',output[0])   # logits [batch,100,2]
  print('output[0].shape:',output[0].shape)
  # output[0].shape: torch.Size([1, 100, 2])

  #print('output[1]:',output[1])   # box  [batch,100,4]
  print('output[1].shape:',output[1].shape)
  # pred_boxes.shape: torch.Size([1,100, 4])

  draw = ImageDraw.Draw(img)
  #pred_logits=output['pred_logits'][0][:, :len(CLASSES)]

  pred_logits=output[0][0]
  print('pred_logits.shape:',pred_logits.shape)
  #pred_logits.shape: torch.Size([100, 2])

  pred_boxes=output[1][0]
  print('pred_boxes.shape:',pred_boxes.shape)
  #pred_boxes.shape: torch.Size([100, 4])


  #print('max_output_test[0][:2]:',max_output_test[0][:4])
  #max_output = pred_logits.softmax(-1).max(-1)
  max_output = torch.max(pred_logits,-1)
  #max_output = pred_logits.max(-1)

  #print('type(max_output):',type(max_output))
  #print('max_output:',max_output)
  print('max_output[0][:2]:',max_output[0][:4]) # max pred
  # max_output[0][:2]: tensor([0.9981, 0.9975, 0.9969, 0.9963], device='cuda:0')
  print('max_output[1][:2]:',max_output[1][:4]) # max class id
  # max_output[1][:2]: tensor([1, 1, 1, 1], device='cuda:0')

  topk = max_output.values.topk(10,largest=False) 
  #print('topk:',topk)
  #topk: torch.return_types.topk(
  # values=tensor([0.7129, 0.7173, 0.7391, 0.7808, 0.8360, 0.9181, 0.9249, 0.9351, 0.9443,0.9524], device='cuda:0'),
  # indices=tensor([71, 59, 92, 98, 72, 45, 87, 75, 19, 90], device='cuda:0'))

  #sys.exit()
  
  #pred_boxes = pred_boxes[topk.indices]

  cnt=0

  #for logits, box in zip(pred_logits, pred_boxes):
  for acc,idx in zip(topk.values,topk.indices):
    box = pred_boxes[idx]
    cls = max_output[1][idx]
    if cls >= len(CLASSES) or cls==0:
      continue
    acc=float(acc)
    print('acc:',acc)
    print('1.0 / acc :',1.0 / acc)
    label = CLASSES[cls]
    print(label)
    #box = box.cpu() * torch.Tensor([800, 600, 800, 600])
    x, y, w,h = box
    x = int(x * float(img.size[0]))
    w = int(w * float(img.size[0]))
    y = int(y * float(img.size[1]))
    h = int(h * float(img.size[1]))

    x0, x1 = x-w//2, x+w//2
    y0, y1 = y-h//2, y+h//2
    draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
    draw.text((x0, y0), label, fill='white')

    break


  img.show()

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

  proc_type=0

  # https://github.com/huggingface/transformers/blob/main/docs/source/ja/model_doc/detr.md
  model = DetrForObjectDetection.from_pretrained(check_point)

  #print(type(model))
  #PATH="./detr-resnet-50_finetuned_zasou-300+100epoch/checkpoint-1600/rng_state.pth"

  #model = torch.load(PATH)
  model.eval()
  model = model.cuda()

  CLASSES = ['N/A', 'zasou']

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    counter += 1

    img=cv2pil(image)

    img_tens = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
      output = model(img_tens)

    if proc_type==0:
      # reffer from
      # /home/nishi/torch_env/lib/python3.10/site-packages/transformers/models/detr/image_processing_detr.py
      #   1503 def post_process(self, outputs, target_sizes):
      out_logits, out_bbox = output.logits, output.pred_boxes
      #print('out_logits:',out_logits)
      #print('out_bbox:',out_bbox)
      prob = torch.nn.functional.softmax(out_logits, -1)
      scores, labels = prob[..., :-1].max(-1)
      #print('scores:',scores)
      #print('labels:',labels)

      results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, out_bbox)]

      idx = results[0]['scores'].argmax()

      acc = results[0]['scores'][idx]
      cls = results[0]['labels'][idx]
      box = results[0]['boxes'][idx]
      label = CLASSES[cls+1]
      acc_f=float(acc)
      if acc_f > thresh:

        print('label:',label)
        print('acc:',acc_f)

        x, y, w,h = box
        x = int(x * float(img.size[0]))
        w = int(w * float(img.size[0]))
        y = int(y * float(img.size[1]))
        h = int(h * float(img.size[1]))

        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2
        acc_s=str(acc_f)
        #draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0))
        #draw.text((x0, y0), label, fill='white')
        cv2.putText(image, label+':'+acc_s, (x0,y0), cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

    else:
      #print('output[0]:',output[0])   # logits [batch,100,2]
      #print('output[0].shape:',output[0].shape)
      # output[0].shape: torch.Size([1, 100, 2])

      #print('output[1]:',output[1])   # box  [batch,100,4]
      #print('output[1].shape:',output[1].shape)
      # pred_boxes.shape: torch.Size([1,100, 4])

      #draw = ImageDraw.Draw(img)
      #pred_logits=output['pred_logits'][0][:, :len(CLASSES)]

      pred_logits=output[0][0]
      #print('pred_logits.shape:',pred_logits.shape)
      #pred_logits.shape: torch.Size([100, 2])

      pred_boxes=output[1][0]
      #print('pred_boxes.shape:',pred_boxes.shape)
      #pred_boxes.shape: torch.Size([100, 4])
      
      #max_output = pred_logits.softmax(-1).max(-1)
      max_output = torch.max(pred_logits,-1)
      #max_output = pred_logits.max(-1)

      #print('type(max_output):',type(max_output))
      #print('max_output:',max_output)
      #print('max_output[0][:2]:',max_output[0][:4]) # max pred
      # max_output[0][:2]: tensor([0.9981, 0.9975, 0.9969, 0.9963], device='cuda:0')
      #print('max_output[1][:2]:',max_output[1][:4]) # max class id
      # max_output[1][:2]: tensor([1, 1, 1, 1], device='cuda:0')

      topk = max_output.values.topk(10,largest=False)
      #print('topk:',topk)
      
      #for logits, box in zip(pred_logits, pred_boxes):
      for acc,idx in zip(topk.values,topk.indices):
        box = pred_boxes[idx]
        cls = max_output[1][idx]
        if cls >= len(CLASSES) or cls==0:
          continue
        acc = float(acc)
        print('acc:',acc)

        if acc < 0.0:
          continue

        print('1.0 / acc :',1.0 / acc)
        label = CLASSES[cls]
        print(label)

        acc_s=str(acc)

        #box = box.cpu() * torch.Tensor([800, 600, 800, 600])
        x, y, w,h = box
        x = int(x * float(img.size[0]))
        w = int(w * float(img.size[0]))
        y = int(y * float(img.size[1]))
        h = int(h * float(img.size[1]))

        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2
        #draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0))

        #draw.text((x0, y0), label, fill='white')
        cv2.putText(image, label+':'+acc_s, (x0,y0), cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)
        break

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