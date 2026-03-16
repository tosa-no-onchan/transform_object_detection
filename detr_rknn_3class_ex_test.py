'''
~/Visualstudio-torch_env/transform_object_detection/
detr_rknn_3class_ex_test.py

'''

import numpy as np
from rknnlite.api import RKNNLite

import cv2
import numpy as np
import time
import glob
from PIL import Image, ImageDraw
from scipy.special import softmax

CLASSES=['zasou','tree','potted-plant','zasou_cluster']


'''
hugging face DTR の学習には、480x480 にアスペクトを維持して、余白を、黒にした画像を
使います。
  padding=True
    画像サイズを、 480x480 にして、余白を埋める。
  padding=False
    アスペクト比を維持して、どちらかが、 480 になるように縮小します。
    画像の余白は、出ません。
'''
def preprocess_universal(img:np.ndarray, target_size=480,padding=True):
    # 1. 画像読み込み (BGR -> RGB)
    #img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    h, w = img.shape[:2]
    # 2. アスペクト比維持のリサイズ (LongestMaxSize相当)
    scale = target_size / max(h, w)     # より rate が小さい方に合わせる
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if not padding:
      return resized
    # 3. パディング (PadIfNeeded / Center相当)
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    # 上下左右に黒帯を追加
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    if False:
        # 4. 正規化 (DETR標準: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225])
        # ※RKNN変換時にモデルに組み込むことも可能ですが、手動なら以下
        input_data = padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_data = (input_data - mean) / std
    else:
       input_data=padded
    # 5. HWC -> CHW 変換 (ONNX/RKNN用)  --> 違う channel last で、OK
    #input_data = input_data.transpose(2, 0, 1)
    # バッチ次元追加
    input_data = np.expand_dims(input_data, axis=0)
    return input_data, (h, w), (pad_h, pad_w), scale

'''
2. 座標の逆算（後処理）
RKNNから出てきた相対座標（0.0~1.0）を、元の画像サイズに戻します。
'''
def postprocess_universal(boxes, orig_shape, pad, scale, target_size=480):
    orig_h, orig_w = orig_shape
    pad_h, pad_w = pad
    # 1. 0~1 の相対座標を 480px 単位に変換
    boxes = boxes * target_size
    # 2. パディング分を差し引く
    boxes[:, [0, 2]] -= pad_w  # xmin, xmax
    boxes[:, [1, 3]] -= pad_h  # ymin, ymax
    # 3. リサイズ倍率で割って元サイズに戻す
    boxes /= scale
    # 4. 画像外にはみ出さないようクリップ
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    return boxes

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
  #rknn.config
  # mean_values=[[123.675, 116.28, 103.53]], # RGB平均値
  # std_values=[[58.395, 57.12, 57.375]],
  # 上記をエミューレートする

  # orginal -> RGB(Pillow)
  x0=x.astype(np.float32) / 255.0  # T.ToTensor() ->  / 255.0
  if rgb==True:
    mean_x=np.array([123.675, 116.28, 103.53],dtype=np.float32)  # RGB?
    std_x =np.array([58.395, 57.12, 57.375],dtype=np.float32)  # RGB ?
  else:
    # [0,2] が入れ替わる
    mean_x=np.array([103.53 , 116.28, 123.675 ],dtype=np.float32)  # BGR ?
    std_x =np.array([57.375 , 57.12, 58.395],dtype=np.float32)  # BGR ?
  return (x0 - mean_x) / std_x

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

def one_shot(img:np.ndarray,rknn_,threshold=0.4):
    hight, width, _ = img.shape
    proc_type=0
    # Hugging Faceで提供されている
    # DETR (DEtection TRansformer) をONNX形式で使用する場合、
    # 入力テンソルのチャンネル順序は RGB です。
    # 一般的なディープラーニングモデルと同様、形状（Shape）は [batch_size, 3, height, width] 
    # の NCHW 形式が標準となります。  --> channel first

    USE_ALL_RKNN_Leave=True
    if USE_ALL_RKNN_Leave:
        # rknn.config で、入力変換を、全て rknn npu に任せる
        # こちらで、全て OK 2026.3.2
        #input_size = (480, 480) 
        #img_resize = cv2.resize(img, input_size)
        #input_data=np.expand_dims(img_resize,axis=0) # batch 軸を、追加
        input_data, (h, w), (pad_h, pad_w), scale = preprocess_universal(img, target_size=480,padding=True)
        #print('input_data.shape',input_data.shape)  # (1, 480, 480, 3)
        #print('input_data.dtype:',input_data.dtype)  # input_data.dtype: uint8
    else:
        # 自分で、入力変換を行う --> いまは、NG
        # transform() の中を、 rknn.config に合わせる
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil=cv2pil(img)   # RGB に変換
        img_resize = image_pil.resize((480, 480))
        #img_resize.show()

        # need resize or not ?
        #np_img = np.array(img_resize)
        np_img = np.array(img_resize)  # RGB
        #print('np_img.shape:',np_img.shape)

        np_img=transform(np_img,rgb=True)
        x2=np.expand_dims(np_img,axis=0)
        # # HWC -> CHW に変更 channel first
        input_data = np.transpose(x2, [0,3,1,2]) 
        #print('input_data.shape',input_data.shape)  # (1, 3, 480, 480)

    # 4. 推論実行
    #start_x = time.time()
    outputs = rknn_lite.inference(inputs=[input_data])
    #print("Inference time:", time.time() - start_x)
    #print("outputs:",outputs)
    if proc_type==0:
        image_pil=cv2pil(img)
        draw = ImageDraw.Draw(image_pil)
        logits = outputs[0]
        bboxes = outputs[1]
        #print('bboxes.shape:',bboxes.shape) # bboxes.shape: (1, 100, 4)
        #print('logits.shape:',logits.shape) # logits.shape: (1, 100, 5)

        prob = softmax(logits, -1)
        # torch 版は、下記で、values と、indices が得られるが、
        # scores, labels = prob[..., :-1].max(-1)
        # こちらは、values のみ
        # なので、indices を自分で加える。
        # 注) 各 logitsの最終column は、 unkonow class なので、使わないみたい。
        scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)

        scores = np.array(scores[0,:])
        idx = scores.argmax()
        #print('idx:',idx)
        score = scores[idx]
        #print('score:',score)

        cls = labels[0][idx]
        #box = bboxes[0][idx]
        boxes=postprocess_universal(bboxes[:,idx], (hight, width), (pad_h, pad_w), scale)
        box = boxes[0]

        label = CLASSES[cls]
        #print('label:',label)
        #print('acc:',float(acc))
        x, y, w,h = box
        #x = int(x * float(image_pil.size[0]))
        #w = int(w * float(image_pil.size[0]))
        #y = int(y * float(image_pil.size[1]))
        #h = int(h * float(image_pil.size[1]))

        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2

        score_s=str(score)
        draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
        draw.text((x0, y0), label+':'+score_s, fill='white')
        #image_pil.show()

        # 2. NumPy配列に変換 (RGB)
        cv2_img = np.array(image_pil)

        # 3. RGBからBGRに変換 (OpenCV用)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Result", cv2_img)

    if proc_type==1:
        image_pil=cv2pil(img)
        draw = ImageDraw.Draw(image_pil)

        #logits = outputs[0][0]  # shape: (100, 2)
        logits = outputs[0]  # shape: (100, 2)
        #bboxes = outputs[1][0]  # shape: (100, 4)
        bboxes = outputs[1]  # shape: (100, 4)

        #print('logits.shape:',logits.shape) # (1, 100, 2)

        prob = softmax(logits, -1)

        # torch 版は、下記で、values と、indices が得られるが、
        # scores, labels = prob[..., :-1].max(-1)
        # こちらは、values のみ
        # なので、indices を自分で加える。
        # 注) 各 logitsの最終column は、 unkonow class なので、使わないみたい。
        scores, labels =prob[..., :-1].max(-1), prob[..., :-1].argmax(-1)

        scores = np.array(scores[0,:])
        idx = scores.argmax()
        #print('idx:',idx)

        score = scores[idx]
        #print('score:',score)

        cls = labels[0][idx]
        box = bboxes[0][idx]

        label = CLASSES[cls]

        #print('label:',label)
        #print('acc:',float(acc))
        x, y, w,h = box
        x = int(x * float(image_pil.size[0]))
        w = int(w * float(image_pil.size[0]))
        y = int(y * float(image_pil.size[1]))
        h = int(h * float(image_pil.size[1]))

        x0, x1 = x-w//2, x+w//2
        y0, y1 = y-h//2, y+h//2
        score_s=str(score)
        draw.rectangle([x0, y0, x1, y1], outline='red', width=1)
        draw.text((x0, y0), label+':'+score_s, fill='white')

        #image_pil.show()

        # 2. NumPy配列に変換 (RGB)
        cv2_img = np.array(image_pil)

        # 3. RGBからBGRに変換 (OpenCV用)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Result", cv2_img)

    if proc_type==2:
        # 2. 推論結果の取得 (outputs = [logits, boxes])
        #print("type(outputs):", type(outputs))  # <class 'list'>
        logits = outputs[0][0]  # shape: (100, 2)
        bboxes = outputs[1][0]  # shape: (100, 4)

        # 3. スコアの計算 (Softmaxを簡易的にシミュレート)
        # logitの[0]が対象スコアの場合。もし逆なら logits[:, 1] に変更してください
        def softmax_x(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        probs = softmax_x(logits)
        scores = probs[:, :-1]  # 対象クラスの確率

        #print("np.max(scores):",np.max(scores))
        threshold=np.max(scores)

        # 4. 描画
        #threshold = 0.5  # 信頼度のしきい値
        #threshold = 0.2  # 信頼度のしきい値
        for i in range(len(scores)):
            if scores[i] >= threshold:
                #print("scores[i]:",scores[i])
                # DETRの出力: [cx, cy, width, height] (0.0~1.0正規化)
                cx, cy, bw, bh = bboxes[i]
                
                # 座標変換: 中心座標(cx, cy) -> 左上(x1, y1), 右下(x2, y2)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                # 枠とスコアを描画
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{scores[i][0]:.2f}", (x1+3 , y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Result", img)


if __name__ == "__main__": 
    # 1. 初期化とモデルのロード
    rknn_lite = RKNNLite()
    # 3class and 余白、黒埋め model
    ret = rknn_lite.load_rknn('./detr_zasou_480_sim.rknn')
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)

    # 2. 推論エンジンの起動 (targetはRK3588)
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime failed')
        exit(ret)

    USE_TEST1=False
    USE_CAMERA=False
    USE_TEST2=True

    if USE_TEST1:
        # 3. ダミー入力データの作成 
        # 変換時に指定した [1, 3, 224, 224] 等のサイズに合わせてください
        #  (,1,3,480,480) か?
        input_data = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)

        # 4. 推論実行
        outputs = rknn_lite.inference(inputs=[input_data])

        # 5. 結果の表示 (最初の数値を少しだけ表示)
        print("Inference success!")
        print("Output type:", type(outputs))
        print("Output shape:", [o.shape for o in outputs])
        print("Result snippet:", outputs[0][0][:5] if len(outputs[0][0]) > 5 else outputs[0])

    if USE_CAMERA:
        cap = cv2.VideoCapture(0)
        #繰り返しのためのwhile文
        cnt=20
        start = time.time()     # [sec]
        while True:
            #カメラからの画像取得
            ret, frame = cap.read()
            #カメラの画像の出力
            #cv2.imshow('camera' , frame)
            one_shot(frame,rknn_lite)
            cnt -= 1
            if cnt <=0: 
                dur = (time.time() - start)/20
                print("fps [Hz]:", 1/dur)
                # fps [Hz]: 8.820546640780357
                cnt=20
                start = time.time()
            #繰り返し分から抜けるためのif文
            key =cv2.waitKey(1)
            if key == 27:
                break

        #メモリを解放して終了するためのコマンド
        cap.release()
        cv2.destroyAllWindows()

    if USE_TEST2:
        #IMG_PATH = "/home/nishi/Documents/VisualStudio-TF/k3_object_detection_using_vision_transformer/101_ObjectCategories/butterfly/"
        IMG_PATH ="../transform_object_detection/datasets/train/zasou/"

        imges=[
            "../transform_object_detection/datasets/train/zasou/10.jpeg",
            "../transform_object_detection/datasets/train/zasou/11.jpeg",
            "../transform_object_detection/datasets/train/zasou/13.jpeg",
            "../transform_object_detection/datasets/train/zasou/12.jpeg",
            "../transform_object_detection/datasets/train/zasou/25.jpeg",
        ]
        files = sorted(glob.glob(IMG_PATH+"*"))

        #for img in imges[:5]:
        for img in files[:10]:
            # 1. 元画像の読み込み (描画用)
            img = cv2.imread(img)
            #h, w, _ = img.shape

            one_shot(img,rknn_lite)
            # 5. 保存
            #cv2.imwrite('result.jpg', img)
            #cv2.imshow("Result", img)
            cv2.waitKey(0)

    # 6. 解放
    rknn_lite.release()