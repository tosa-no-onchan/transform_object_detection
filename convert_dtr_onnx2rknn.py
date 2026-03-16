'''
convert_dtr_onnx2rknn.py

この前に
1.  saved model -> ONNX へ変換と slim 化は、下記に記述しました。
zasou_dtr2onnx.py

2. onnx -> rknn 変換
(rknn_env310)
$ python convert_dtr_onnx2rknn.py

'''
from rknn.api import RKNN
import os

USE_3class_ex=True
USE_pixel_mask=False

#ONNX_MODEL = 'torch_model.onnx'
# zasou_train_3class_ex.ipynb で作った model だと、
#注) ターミナルから、下記 cli を行う。
# $ optimum-cli export onnx --model detr-resnet-50_finetuned_zasou_ex/final_model --task object-detection onnx/
# $ pip install onnxsim
# $ onnxsim detr_zasou_480.onnx detr_zasou_480_sim.onnx
# 入力サイズの固定化が必要かも!!
# $ onnxsim detr_zasou_480.onnx detr_zasou_480_sim.onnx --overwrite-input-shape pixel_values:1,3,480,480

if USE_3class_ex:
    ONNX_MODEL = 'detr_zasou_480_sim.onnx'
    RKNN_MODEL = 'detr_zasou_480_sim.rknn'
else:
    ONNX_MODEL = 'model.onnx'
    RKNN_MODEL = 'torch_model.rknn'

USE_CONTIZE=True
rknn = RKNN(verbose=True)

# 全て npu にまかせる  --> こちらで、good!
# dtr_rknn_test.py で、
# USE_ALL_RKNN_Leave=True に、する事
if True:
    # v2.3.2 での推奨設定
    if USE_CONTIZE:
        rknn.config(
            target_platform='rk3588',
            # DETR (ImageNet基準) の正規化設定をそのまま残す
            mean_values=[[123.675, 116.28, 103.53]], # RGB平均値 (255倍した値)
            std_values=[[58.395, 57.12, 57.375]],    # (255倍した値)
            optimization_level=3,
            # quantized_dtype は指定しなくてOK（do_quantization=Falseなら無視されます）
            quantized_dtype='asymmetric_quantized-8', # Transformerは非対称量子化が精度的に有利な場合が多い
        )
    else:
        rknn.config(
            target_platform='rk3588',
            # DETR (ImageNet基準) の正規化設定をそのまま残す
            mean_values=[[123.675, 116.28, 103.53]],
            std_values=[[58.395, 57.12, 57.375]],
            optimization_level=3,
            # quantized_dtype は指定しなくてOK（do_quantization=Falseなら無視されます）
        )

# こちらが、 onnx 版の入力処理のエミューレート版を使う場合
# 自分で、 input を加工する場合は、こちら。ただし、今は、精度が落ちるので、上記を使うこと
# USE_ALL_RKNN_Leave=False に、する事
else:
    if USE_CONTIZE:
        rknn.config(
            target_platform='rk3588',
            optimization_level=3,
            # mean_values, std_values, reorder_channel は書かない
            # これにより、入力された値（0.0〜1.0）がそのままモデルに渡されます
            quantized_dtype='asymmetric_quantized-8', # Transformerは非対称量子化が精度的に有利な場合が多い
            # quantized_algorithm='normal', 
        )
    else:
        rknn.config(
            target_platform='rk3588',
            optimization_level=3,
            # mean_values, std_values, reorder_channel は書かない
            # これにより、入力された値（0.0〜1.0）がそのままモデルに渡されます
        )

# 2. ONNXロード
#ret = rknn.load_onnx(model=ONNX_MODEL)
if USE_pixel_mask:
    # もしONNXにpixel_maskが含まれている場合  --> いまは、NG
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        inputs=['pixel_values', 'pixel_mask'],
        input_size_list=[[1, 3, 480, 480], [1, 480, 480]]
    )
else:
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        inputs=['pixel_values'],
        #input_size_list=[[1, 3, 224, 224]] # モデルの想定サイズ（例: 224x224）に合わせる
        input_size_list=[[1, 3, 480, 480]] # モデルの想定サイズ（例: 480x480）に合わせる
    )

if ret != 0:
    print('Load ONNX failed!')
    exit(ret)

#量子化の場合
if USE_CONTIZE:
    #image_dir = './calibration_images'
    # zasou_train_3class_ex.ipynb で作った、onnx だと、
    # 480x480 にリサイズして、余白を、黒で埋めた、test用 画像 が必要
    #IMG_PATH ="../transform_object_detection/datasets/train/zasou"
    IMG_PATH ="/home/nishi/Documents/VisualStudio-TF/annotation/data-backup/雑草"
    is_file = os.path.isfile('dataset.txt')
    if not is_file:
        print("make dataset.txt")
        with open('dataset.txt', 'w') as f:
            for img_name in os.listdir(IMG_PATH):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    f.write(os.path.join(IMG_PATH, img_name) + '\n')
        print("done!")

if USE_CONTIZE:
    # 量化を有効にしてビルド
    ret = rknn.build(
        do_quantization=True, 
        dataset='./dataset.txt'  # ここでファイルパスを指定
    )
else:
    # 3. build（量子化なし）
    ret = rknn.build(do_quantization=False)

if ret != 0:
    print('Build failed!')
    exit(ret)

# 4. RKNN出力
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn failed!')
    exit(ret)

print('Done')
rknn.release()

