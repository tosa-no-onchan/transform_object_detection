#### transform_object_detection  

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

  雑草データセットでの学習  
  zasou_train.ipynb 　

  検証  
  zasou_Inference.ipynb  

元の記事  
  [Transformers Object detection を試す。](http://www.netosa.com/blog/2024/07/transformers-object-detection.html)  
