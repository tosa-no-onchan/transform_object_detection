import numpy as np
import os
from PIL import Image, ImageDraw
import sys

import datasets
# datsets.Featuresを定義
my_features = datasets.Features(
{ 'image': datasets.Image(mode=None, decode=True, id=None),
 'img_sub_path': datasets.Value(dtype='string', id=None),
 'image_id': datasets.Value(dtype='int64', id=None),
 'width': datasets.Value(dtype='int32', id=None),
 'height': datasets.Value(dtype='int32', id=None),
 'objects': datasets.Sequence(feature={'id': datasets.Value(dtype='int64', id=None), 'area': datasets.Value(dtype='int64', id=None), 'bbox': datasets.Sequence(feature=datasets.Value(dtype='float32', id=None), length=4, id=None), 
                                       'category': datasets.ClassLabel(names=['zasou', 'tree', 'potted_plant'], id=None)}, length=-1, id=None)})


from datasets import load_dataset
#dataset = load_dataset("imagefolder", data_dir="datasets")
dataset = load_dataset("imagefolder", data_dir="datasets", features=my_features)

print('dataset:',dataset)
#dataset: DatasetDict({
#    train: Dataset({
#        features: ['image', 'img_sub_path', 'image_id', 'width', 'height', 'objects'],
#        num_rows: 314
#    })
#})

print('dataset[\'train\'][2]',dataset['train'][2])
#dataset['train'][2] {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FBD7183DD90>, 'img_sub_path': 'zasou', 'image_id': 801, 'width': 640, 'height': 480, 'objects': {'id': [2], 'area': [271122], 'bbox': [[6.0, 30.0, 619.0, 438.0]], 'category': [0]}}


from transformers import AutoImageProcessor
checkpoint = "facebook/detr-resnet-50"
#checkpoint = "./detr-resnet-50_finetuned_zasou-2024.7.7-2"
#checkpoint = "./detr-resnet-50_finetuned_zasou-300epoch"
#最新の高速版は、こちら
#image_processor = AutoImageProcessor.from_pretrained(checkpoint)
# 従来版は、こちら
image_processor = AutoImageProcessor.from_pretrained(checkpoint,use_fast=False)


# 画像の拡張
import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,     # int
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)
    return annotations


# transforming a batch
def transform_aug_ann(examples):
    print("type:", type(examples))
    # type: <class 'dict'>
    print("examples.keys():", examples.keys())
    #examples.keys(): dict_keys(['image', 'img_sub_path', 'image_id', 'width', 'height', 'objects'])
    sys.exit()

    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]
    return image_processor(images=images, annotations=targets, return_tensors="pt")


dataset['train'] = dataset['train'].with_transform(transform_aug_ann)
#dataset['train'][15]

import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from torchvision.transforms.functional import pil_to_tensor, to_pil_image
#example = dataset['train'][15]


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

#id2label = {0: 'zasou'}
id2label={0:'zasou',1:'tree',2:'potted_plant'}
#label2id = {'zasou': 0}
label2id={'zasou':0,'tree':1,'potted_plant':2}

if False:
    # または属性から直接確認
    #print(dataset.num_rows)   # {'train': 264}
    print(dataset["train"].num_rows)  # 264
    rows=dataset["train"].num_rows
    cat_list=[]
    for i in range(rows):
        category=dataset['train'][i]['objects']['category']
        for cat in category:
            print('cat',cat)
            #if not cat_list[cat]:
            #    cat_list[cat]=cat
        break


from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments
USE_ARG1=False
USE_ARG2=False
USE_ARG_ORG=True
if USE_ARG1:
    # こちらは、検証用データが必要
    training_args = TrainingArguments(
        output_dir="detr-resnet-50_finetuned_zasou",
        per_device_train_batch_size=8,
        num_train_epochs=180,
        fp16=True,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=True,
        # --- 追加/変更を検討したい項目 ---
        #evaluation_strategy="epoch",     # エポックごとに評価
        eval_strategy="epoch",       # ← ここを eval_strategy に変更
        save_strategy="epoch",       # (こちらはそのままでも動きますが、推奨は epoch です)
        load_best_model_at_end=True,     # 最良モデルを最後に読み込む
        logging_first_step=True,
        lr_scheduler_type="cosine",      # 学習率を徐々に下げる（精度が安定しやすい）
    )
if USE_ARG2:
    training_args = TrainingArguments(
        output_dir="detr-resnet-50_finetuned_zasou",
        per_device_train_batch_size=8,
        num_train_epochs=180,
        fp16=True,
        learning_rate=1e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=True,
        # --- 追加/変更を検討したい項目 ---
        #evaluation_strategy="epoch",     # エポックごとに評価
        eval_strategy="no",       # ← ここを eval_strategy に変更
        save_strategy="epoch",       # (こちらはそのままでも動きますが、推奨は epoch です)
        load_best_model_at_end=False,     # 最良モデルを最後に読み込む
        logging_first_step=True,
        lr_scheduler_type="cosine",      # 学習率を徐々に下げる（精度が安定しやすい）
    )
if USE_ARG_ORG:
    training_args = TrainingArguments(
        output_dir="detr-resnet-50_finetuned_zasou",
        per_device_train_batch_size=8,
        num_train_epochs=180,
        fp16=True,
        learning_rate=1e-5,
        #learning_rate=1.875e-8,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=True,
        #warmup_steps=500,
        #save_steps=200,
        save_strategy="epoch",       # (こちらはそのままでも動きますが、推奨は epoch です)
        logging_steps=200,
        logging_first_step=True,
        lr_scheduler_type="cosine",      # 学習率を徐々に下げる（精度が安定しやすい）
    )


from transformers import Trainer
if USE_ARG1:
    #print(dataset.num_rows)   # {'train': 264}
    rows=dataset.num_rows['train']
    ratio = 0.8  # 8:2に分ける
    
    # 分割する位置を計算（整数にする必要あり）
    split_point = int(rows * ratio)
    
    # スライスで分割
    train_dt = dataset["train"][:split_point]  # 0からsplit_pointの手前まで
    test_dt = dataset["train"][split_point:]   # split_pointから最後まで
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dt,
        eval_dataset=test_dt,
        # tokenizer=image_processor,  <-- これを削除
        processing_class=image_processor, # <-- これに変更
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        # tokenizer=image_processor,  <-- これを削除
        processing_class=image_processor, # <-- これに変更
    )
    

trainer.train()

