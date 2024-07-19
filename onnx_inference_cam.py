# -*- coding: utf-8 -*-
"""
onnx_inference_cam.py

Visualstudio-torch_env/transform_object_detection/onnx_inference_cam.py

https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/simple_onnxruntime_inference.ipynb
"""

import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example

import os
import sys
import time

import cv2

from PIL import Image, ImageDraw

from scipy.special import softmax


example_model = get_example("/home/nishi/Documents/Visualstudio-torch_env/transform_object_detection/onnx/model.onnx")
CLASSES=['none','zasou']

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
# softmax(x) = np.exp(x)/sum(np.exp(x))
"""
import numpy as np
from scipy.special import softmax

np.set_printoptions(precision=5)
m = softmax(x)
"""

def get_pred(pred_logits, pred_boxes,thresh):
  pred_results=[]
  for logits, box in zip(pred_logits[0], pred_boxes[0]):
    cls = softmax(logits).argmax()
    #cls = logits.argmax()
    #print('cls:',cls)
    if cls >= len(CLASSES) or cls == 0:
      continue
    pred = 1.0 / logits[cls]
    if pred < thresh:
      continue
    pred_results.append([cls,pred,box])
  return pred_results


# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#Normalize
# def __init__(self, mean, std, inplace=False):
# (input[channel] - mean[channel]) / std[channel]
#transform = T.Compose([
#    T.ToTensor(),
#    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])
def transform(x,rgb=True):
  # orginal -> RGB(Pillow)
  x0=x.astype(np.float32) / 255.0  # T.ToTensor() ->  / 255.0
  if rgb==True:
    mean_x=np.array([0.485, 0.456, 0.406],dtype=np.float32)  # RGB?
    std_x =np.array([0.229, 0.224, 0.225],dtype=np.float32)  # RGB ?
  else:
    mean_x=np.array([0.406, 0.456, 0.485],dtype=np.float32)  # BGR ?
    std_x =np.array([0.225, 0.224, 0.229],dtype=np.float32)  # BGR ?
  return (x0 - mean_x) / std_x

def one_shot():
  path='./datasets/train/zasou'
  flist=os.listdir(path)
  cnt=0

  proc_type=0

  image = Image.open(path+'/'+flist[cnt])
  cnt+=1
  #image.show()
  #print('image.type:',type(image))

  print('image.size:',image.size)

  img_resize = image.resize((480, 480))
  #img_resize.show()

  # need resize or not ?
  #np_img = np.array(img_resize)
  np_img = np.array(image)  # BGR
  print('np_img.shape:',np_img.shape)

  np_img=transform(np_img)

  #print('np_img:',np_img)
  #print('np_img.dtype:',np_img.dtype)

  # https://book-read-yoshi.hatenablog.com/entry/2021/06/07/tensorflow_pytorch_image_data_channel_first_last_change

  x2=np.expand_dims(np_img,axis=0)
  x = np.transpose(x2, [0,3,1,2])

  print('x.shape:',x.shape)

  #x = x.astype(np.float32)

  # torch 版は、下記処理で、ノーマライズしているので、こちらも必要だが!!
  #   img_tens = transform(img).unsqueeze(0).cuda()

  logits,pred_boxes = sess.run([output_name, output_name_1], {input_name: x})

  print(len(logits[0]))

  batch_id=0
  #print('pred_logits:',pred_logits[batch_id])
  #print("")
  print(len(pred_boxes[0]))
  #print('pred_boxes:',pred_boxes[batch_id])

  num_queries=len(logits[batch_id])
  print('num_queries:',num_queries)

  # https://www.geeksforgeeks.org/object-detection-with-detection-transformer-dert-by-facebook/

  draw = ImageDraw.Draw(image)
  thresh=0.7

  if proc_type==0:
    # reffer from
    # /home/nishi/torch_env/lib/python3.10/site-packages/transformers/models/detr/image_processing_detr.py
    #   1503 def post_process(self, outputs, target_sizes):

    #print('logits:',logits)
    prob = softmax(logits, -1)
    #print('type(prob):',type(prob))
    print('prob.shape:',prob.shape)
    #print('prob:',prob)
    #print('prob[..., :-1]:',prob[..., :-1])

    # torch 版は、下記で、values と、indices が得られるが、
    # scores, labels = prob[..., :-1].max(-1)
    # こちらは、values のみ
    # なので、indices を自分で加える。
    scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)

    #print('scores:',scores)
    #print('labels:',labels)

    #print('logits:',logits)
    scores = np.array(scores[0,:])
    idx = scores.argmax()

    print('idx:',idx)

    score = scores[idx]

    print('score:',score)

    cls = labels[0][idx]
    box = pred_boxes[0][idx]

    label = CLASSES[cls+1]
    #print('label:',label)
    #print('acc:',float(acc))
    x, y, w,h = box
    x = int(x * float(image.size[0]))
    w = int(w * float(image.size[0]))
    y = int(y * float(image.size[1]))
    h = int(h * float(image.size[1]))

    x0, x1 = x-w//2, x+w//2
    y0, y1 = y-h//2, y+h//2
    score_s=str(score)
    draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
    draw.text((x0, y0), label+':'+score_s, fill='white')

    image.show()

    sys.exit()


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

  proc_type=0

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
    counter+=1
    #imagex=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_pil=cv2pil(image)

    #print('image_pil.shape:',image_pil.shape)

    np_img = np.array(image_pil)
    np_img=transform(np_img)

    # https://book-read-yoshi.hatenablog.com/entry/2021/06/07/tensorflow_pytorch_image_data_channel_first_last_change

    x2=np.expand_dims(np_img,axis=0)
    x = np.transpose(x2, [0,3,1,2])

    # ['batch_size', 'num_channels', 'height', 'width']
    #print('x.shape:',x.shape)

    #x = x.astype(np.float32)

    logits,pred_boxes = sess.run([output_name, output_name_1], {input_name: x})
    #print(len(logits[0]))


    if proc_type==0:
      # reffer from
      # /home/nishi/torch_env/lib/python3.10/site-packages/transformers/models/detr/image_processing_detr.py
      #   1503 def post_process(self, outputs, target_sizes):

      #print('logits:',logits)
      #prob = torch.nn.functional.softmax(logits, -1)
      prob = softmax(logits, -1)

      # torch 版は、下記で、values と、indices が得られるが、
      # scores, labels = prob[..., :-1].max(-1)
      # こちらは、values のみ
      # なので、indices を自分で加える。
      scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)

      scores = np.array(scores[0,:])
      idx = scores.argmax()

      #print('idx:',idx)
      score = scores[idx]

      print('score:',score)
      if score > thresh:

        cls = labels[0][idx]
        box = pred_boxes[0][idx]

        label = CLASSES[cls+1]
        #print('label:',label)
        #print('acc:',float(acc))
        x, y, w,h = box
        #print('x:',x)
        #print('type)x:',type(x))
        x *= float(image.shape[0])
        w *= float(image.shape[0])
        y *= float(image.shape[1])
        h *= float(image.shape[1])

        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2
        x0=int(x0)
        x1=int(x1)
        y0=int(y0)
        y1=int(y1)
        score_s=str(score)
        #draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
        #draw.text((x0, y0), label+':'+score_s, fill='white')
        cv2.rectangle(image, (y0, x0), (y1, x1), (255, 0, 0))
        cv2.putText(image, label+':'+score_s, (y0, x0), cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

    detection_result_image=image

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

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

  providers = ["CUDAExecutionProvider"]

  if True:
    sess = onnxruntime.InferenceSession(example_model)
  else:
    sess = onnxruntime.InferenceSession(example_model, providers=providers)

  input_name = sess.get_inputs()[0].name
  print('input_name:',input_name)
  # 'pixel_values'
  input_shape = sess.get_inputs()[0].shape
  print('input_shape:',input_shape)
  # ['batch_size', 'num_channels', 'height', 'width']
  input_type = sess.get_inputs()[0].type
  print('input_type:',input_type)
  # 'tensor(float)'

  outputs = sess.get_outputs()
  print('outputs:',outputs)

  output_name = sess.get_outputs()[0].name
  print('output_name[0]:',output_name)
  # 'logits'
  output_name_1 = sess.get_outputs()[1].name
  print('output_name[1]:',output_name_1)
  # output_name[1]: pred_boxes

  output_shape = sess.get_outputs()[0].shape
  print('output_shape:',output_shape)
  # ['batch_size', 'num_queries', 2]
  output_shape_1 = sess.get_outputs()[1].shape
  print('output_shape_1:',output_shape_1)
  # output_shape_1: ['batch_size', 'num_queries', 4]

  output_type = sess.get_outputs()[0].type
  print('output_type:',output_type)
  # 'tensor(float)'
  output_type_1 = sess.get_outputs()[1].type
  print('output_type_1:',output_type_1)


  if True:
    cam_movie()
  else:
    one_shot()

