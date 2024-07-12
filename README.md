#### transform_object_detection  

  [Transformers Object Detection](https://huggingface.co/docs/transformers/v4.42.0/ja/tasks/object_detection) を参考に、雑草データセットを転移学習させてみた。    

  google の雑草画像のスクレイピングを行う。  
  [tosa-no-onchan/annotation](https://github.com/tosa-no-onchan/annotation)  

  雑草画像を、YOLO annotaion にする。  

  YOLO annotation から、COCO huggingface metadata の JSONL file を作成する。  
  $ python yolo2huggingface_metadata.py  

  雑草データセットでの学習  
  zasou_train.ipynb 
