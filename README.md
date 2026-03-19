#### transform_object_detection  
  Hugging Face detr (object detection transformer)  
  [Transformers Object Detection](https://huggingface.co/docs/transformers/v4.42.0/ja/tasks/object_detection) を参考に、雑草データセットを転移学習させてみた。    

  google の雑草画像のスクレイピングを行う。  
  [tosa-no-onchan/annotation](https://github.com/tosa-no-onchan/annotation)  
  使える画像だけ残して、使えない画像は、削除する。  

  上記、雑草画像を、YOLO annotaion にする。  
  labelImg で、YOLO形式のアノテーションにする。  
  class id and Name  0:"zasou"  

  YOLO annotation から、COCO huggingface metadata の JSONL file を作成する。  
  $ python yolo2huggingface_metadata.py  
  ./datasets/train/zasou/metadata.jsonl  

##### 1. 雑草データセットでの学習(旧版)  
  zasou_train.ipynb 　

##### 2. 検証(旧版)  
  zasou_Inference.ipynb  
  
##### 3. 雑草　3class 学習  最新版なので、こちらを使って!! by nishi 2026.3.19  
  i) 入力画像を、 アスペクト比をそのままに、480x480 にリサイズして、余白には、黒(0,0,0) を埋め込みます。  
  ii) def transform_aug_ann(examples) のバグも改修しています。  
  iii) input size 480x480 にしています。  
````
    image_processor = AutoImageProcessor.from_pretrained(  
      img_checkpoint,   
      size={"shortest_edge": 480, "longest_edge": 1333},  
      use_fast=False # 必要に応じて  
    )  
````
  zasou_train_3class.ipynb  

####  4. 雑草　3class 検証  
  zasou_Inference2_ex.ipynb

元の記事  
  [Transformers Object detection - detr の転移学習とONNX変換と実行。](http://www.netosa.com/blog/2024/07/transformers-object-detection.html)  
  [Transformers Object detection - detr の転移学習とONNX変換と実行。#2](https://www.netosa.com/blog/2026/03/transformers-object-detection---detr-onnx2.html)  
