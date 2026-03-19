'''
zasou_dtr2onnx.py

DETRをONNXへエクスポートする際、
「入力サイズを480x480に固定」し、かつ「後処理（Post-processing）を切り離す」ことが、その後のRKNN変換を成功させる鍵になります。
以下のコードで、RKNNに最適化されたONNXファイルを出力できます。

注1) 
opset_version=13 で、OK です!!

1. saved model の ONNX へ変換
$ python zasou_dtr2onnx.py
  --> detr_zasou_480.onnx

1.1 CLI で変換(使わなくても、OK)
$ optimum-cli export onnx --model detr-resnet-50_finetuned_zasou_ex/final_model --task object-detection onnx/

2. 書き出し後の重要ステップ（ONNX Simplifier）
DETRのONNXグラフは非常に複雑で、そのままではRKNN変換でエラーが出やすいです。必ず onnxsim を使ってグラフを簡略化してください。

# ターミナルで実行
$ pip install onnxsim
$ onnxsim detr_zasou_480.onnx detr_zasou_480_sim.onnx

入力サイズの固定化が必要!!
$ onnxsim detr_zasou_480.onnx detr_zasou_480_sim.onnx --overwrite-input-shape pixel_values:1,3,480,480
$ onnxsim onnx/model.onnx detr_zasou_480_sim.onnx --overwrite-input-shape pixel_values:1,3,480,480
  --> detr_zasou_480_sim.onnx

3. onnx -> rknn
(rknn_env310) で、行うこと

(rknn_env310) $ python convert_dtr_onnx2rknn.py

'''

import torch
from transformers import AutoModelForObjectDetection

# 1. 学習済みモデルのロード
#model_path = "./detr-resnet-50_finetuned_zasou_ex/final_model"
model_path = "./detr-resnet-50_finetuned_zasou/final_model"
model = AutoModelForObjectDetection.from_pretrained(model_path)
model.eval()

# 2. ダミー入力の作成 (バッチサイズ1, 3チャンネル, 480x480)
# RKNN用にサイズを固定します
pixel_values = torch.randn(1, 3, 480, 480)
# 全域が画像であることを示すマスク（全て1）
pixel_mask = torch.ones(1, 480, 480, dtype=torch.long)

# 3. エクスポート実行
onnx_model_path = "detr_zasou_480.onnx"
USE_ORG=True
USE_SDP=False
USE_pixel_mask=False

# 下記で、OK です。 by nishi 2026.3.15
if USE_ORG:
  torch.onnx.export(
      model,
      #(pixel_values, pixel_mask), # 入力タプル
      (pixel_values), # pixel_mask を渡さない
      onnx_model_path,
      export_params=True,        # 重みをファイルに書き込む
      opset_version=13,          # DETRの演算をサポートするバージョン
      do_constant_folding=True,  # 定数畳み込みでグラフを最適化
      input_names=['pixel_values'],
      output_names=['logits', 'pred_boxes'], # クラススコアと座標の出力名
      # dynamic_axes はあえて指定せず、サイズを 480x480 に固定します（RKNN向け）
  )

# 下記も、 OK です。
if USE_SDP:
  import torch.nn.functional as F
  # 1. SDPA（問題の演算）を無効化し、旧来の計算方式を強制する
  with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
      # このコンテキストの中でエクスポートを実行
      torch.onnx.export(
          model,
          #(pixel_values, pixel_mask),
          (pixel_values),
          onnx_model_path,
          export_params=True,
          opset_version=13,  # エラーの指示通り14に上げます
          do_constant_folding=True,
          #input_names=['pixel_values', 'pixel_mask'],
          input_names=['pixel_values'],
          output_names=['logits', 'pred_boxes'],
      )

# こいつは、NG みたい!!
if USE_pixel_mask:
  torch.onnx.export(
      model,
      (pixel_values, pixel_mask), # 入力タプル
      onnx_model_path,
      export_params=True,        # 重みをファイルに書き込む
      opset_version=13,          # DETRの演算をサポートするバージョン
      do_constant_folding=True,  # 定数畳み込みでグラフを最適化
      input_names=['pixel_values', 'pixel_mask'],
      output_names=['logits', 'pred_boxes'], # クラススコアと座標の出力名
      # dynamic_axes はあえて指定せず、サイズを 480x480 に固定します（RKNN向け）
  )

print(f"ONNX model saved to {onnx_model_path}")

