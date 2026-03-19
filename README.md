### transform_object_detection  
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

#### 1. 雑草データセットでの学習(旧版)  
  zasou_train.ipynb 　

#### 2. 検証(旧版)  
  zasou_Inference.ipynb  
  
#### 3. 雑草　3class 学習  最新版なので、こちらを使って!! by nishi 2026.3.19  
  zasou_train_3class.ipynb  

  i) 入力画像を、 アスペクト比をそのままに、480x480 にリサイズして、余白には、黒(0,0,0) を埋め込みます。  
````
USE_ASPECT_FIX=True
if USE_ASPECT_FIX:
    # アスペクト比を維持させる
    transform = albumentations.Compose(
        [
            # 1. アスペクト比を維持し、長い方の辺を480ピクセルに合わせる
            albumentations.LongestMaxSize(max_size=480),
            # 2. 足りない部分を黒（0）で埋めて 480x480 に固定する
            albumentations.PadIfNeeded(
                min_height=480, 
                min_width=480, 
                border_mode=0, # 定数（黒）で埋める
                #value=(0, 0, 0)
                fill=0  # 'value' を 'fill' に変更（黒にする場合は 0 または [0, 0, 0]）
            ),
            # 3. その他のデータ拡張
            albumentations.HorizontalFlip(p=0.5),
        ],
        # bboxもパディングに合わせて自動で座標調整されます
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
    )
````
  
  ii) def transform_aug_ann(examples) のバグも改修しています。  
````
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        #image = np.array(image.convert("RGB"))[:, :, ::-1]   # NG
        # 1. RGBのままNumPy化（[::-1] は削除！）
        image_np = np.array(image.convert("RGB"))
        # 2. Albumentations実行 (480x480にパディング)
        out = transform(image=image_np, bboxes=objects["bbox"], category=objects["category"])
        area.append(objects["area"])
        images.append(out["image"])    # リストに追加
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]
    # 3. images(リスト)を渡し、size指定は定義時のものが使われるため省略可
    return image_processor(images=images, annotations=targets, return_tensors="pt")

````
  iii) input size 480x480 にしています。  
````
    image_processor = AutoImageProcessor.from_pretrained(  
      img_checkpoint,   
      size={"shortest_edge": 480, "longest_edge": 1333},  
      use_fast=False # 必要に応じて  
    )  
````

#### 4. 雑草　3class 検証  
  zasou_Inference2_ex.ipynb

#### 5. YOLO アノテーション to metadata.jsonl  
  $ python yolo2huggingface_metadata2.py

#### 6. torch 転移学習済モデルを onnx に変換  
  $ python zasou_dtr2onnx.py  

#### 7. onnx のスリム化と入力サイズの Fix  
  $ onnxsim detr_zasou_480.onnx detr_zasou_480_sim.onnx --overwrite-input-shape pixel_values:1,3,480,480  

#### 8. onnx -&gt; rknn 変換  
  (rknn_env310) $ python convert_dtr_onnx2rknn.py  


元の記事  
  [Transformers Object detection - detr の転移学習とONNX変換と実行。](http://www.netosa.com/blog/2024/07/transformers-object-detection.html)  
  [Transformers Object detection - detr の転移学習とONNX変換と実行。#2](https://www.netosa.com/blog/2026/03/transformers-object-detection---detr-onnx2.html)  
