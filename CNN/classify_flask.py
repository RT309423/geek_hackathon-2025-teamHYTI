# classify_flask.py

import torch
# V の定義を追加 (torch.autograd.Variable のエイリアスとして)
from torch.autograd import Variable as V 
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify # Flask関連をインポート


# --- 追加ここから ---
from flask_cors import CORS # CORS対応のために追加
# --- 追加ここまで ---

# AIモデルのロード、画像変換、カテゴリ情報の関数化 (サーバー起動時に一度だけ実行)
def setup_ai_model():
    print("AIモデルをロードしています...少々お待ちください。")

    arch = 'resnet18'
    # モデルファイルのパスをスクリプトからの相対パスで指定
    # os.path.dirname(__file__) は現在のスクリプトがあるディレクトリのパスを取得します
    model_file = os.path.join(os.path.dirname(__file__), 'resnet18_places365.pth.tar')
    
    # モデルのロード
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval() # 推論モードに設定

    # 画像変換の設定
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 365カテゴリのラベル情報をロード
    # カテゴリファイルのパスをスクリプトからの相対パスで指定
    file_name = os.path.join(os.path.dirname(__file__), 'categories_places365.txt')
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])

    # カテゴリのマッピング辞書
    category_mapping = {
        # 「家」に分類されるもの
        "attic": "家", "balcony/interior": "家", "bathroom": "家", "bedroom": "家", "childs_room": "家", 
        "closet": "家", "corridor": "家", "dining_room": "家", "home_office": "家", "kitchen": "家", 
        "living_room": "家", "pantry": "家",
        # 「街中」に分類されるもの
        "alley": "街中", "apartment_building/outdoor": "街中", "building_facade": "街中", "crosswalk": "街中", 
        "downtown": "街中", "skyscraper": "街中", "street": "街中",
        # 「海」に分類されるもの
        "beach": "海", "coast": "海", "ocean": "海", "wave": "海", "lagoon": "海",
        # 「山（森）」に分類されるもの
        "bamboo_forest": "山（森）", "canyon": "山（森）", "forest/broadleaf": "山（森）",
        "mountain": "山（森）", "mountain_path": "山（森）",
        # 「公園」に分類されるもの
        "botanical_garden": "公園", "campsite": "公園", "playground": "公園",
        # 「学校」に分類されるもの
        "campus": "学校", "classroom": "学校", "lecture_room": "学校", "library/indoor": "学校"
    }
    
    print("AIモデルのロードが完了しました。")
    return model, centre_crop, classes, category_mapping

# AIモデルと関連設定をロード (サーバー起動時に一度だけ実行)
model, centre_crop, classes, category_mapping = setup_ai_model()


# サーバーが最後に識別した結果を保持するグローバル変数
# 初期値は「未分類」「不明」としておく
last_classification_result = {"place": "未分類", "time": "不明"}

# Python側で識別する固定画像のパス
# classify_flask.pyと同じディレクトリにあるmy_photo.jpgを指すようにします
FIXED_IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'my_photo.jpg')

# 画像を自動的に識別し、結果を更新する関数
def process_fixed_image_periodically():
    global last_classification_result # グローバル変数を更新するために必要
    
    # 固定画像の存在チェック
    if not os.path.exists(FIXED_IMAGE_PATH):
        print(f"警告: 固定画像 '{FIXED_IMAGE_PATH}' が見つかりません。")
        last_classification_result = {"place": "画像パスエラー", "time": "不明"}
        return # ここで処理を終了

    print(f"固定画像 '{FIXED_IMAGE_PATH}' を識別します...")
    try:
        img = Image.open(FIXED_IMAGE_PATH)
        result = process_image(img) # 既存のprocess_image関数を呼び出す
        last_classification_result = result
        print(f"固定画像の識別完了。結果: {result}")
    except Exception as e:
        print(f"固定画像識別中にエラーが発生しました: {e}")
        last_classification_result = {"place": "識別エラー", "time": "不明"}


def analyze_time_of_day(image_object):
    """
    Pillowの画像オブジェクトを受け取り、明るさと色温度から時間帯を判定する関数
    戻り値: "日中", "夕方", "夜"
    """
    # 画像をnumpy配列に変換し、HSV色空間（色相・彩度・明度）に変換
    img_hsv = image_object.convert('HSV')
    img_np = np.array(img_hsv)

    # 明度(Value)の平均値を計算 (0-255の範囲)
    avg_brightness = np.mean(img_np[:, :, 2])

    # 色相(Hue)のヒストグラムを計算
    hue_hist = np.histogram(img_np[:, :, 0], bins=18, range=(0, 255))[0]

    # 判定ロジック（しきい値は調整可能）
    BRIGHTNESS_NIGHT = 60    # この明るさより暗ければ「夜」
    BRIGHTNESS_EVENING = 120 # この明るさより暗く、オレンジ色が多ければ「夕方」

    # まず明るさで「夜」を判定
    if avg_brightness < BRIGHTNESS_NIGHT:
        return "夜"

    # 「夕方」かどうかを色で判定
    # オレンジ〜赤色の範囲の色相の割合を計算 (PillowのHSVではオレンジ/赤はおおよそ0-30の範囲)
    orange_red_ratio = np.sum(hue_hist[0:3]) / np.sum(hue_hist)
    if avg_brightness < BRIGHTNESS_EVENING and orange_red_ratio > 0.3: # 30%以上が暖色なら
        return "夕方"
    
    # 上記以外は「日中」と判定
    return "日中"

def process_image(image_object):
    """
    Pillowの画像オブジェクトを受け取り、場所と時間帯を分析して辞書で返す
    """
    # 場所の分類
    input_img = V(centre_crop(image_object).unsqueeze(0))
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    predicted_by_model = classes[idx[0]]
    final_category = category_mapping.get(predicted_by_model, "その他")

    # 時間帯の分類
    time_of_day = analyze_time_of_day(image_object)

    return {"place": final_category, "time": time_of_day}


# --------------------------------------------------------------------------
# Flaskアプリケーションの定義
# --------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    global last_classification_result # グローバル変数を更新するために必要

    if 'image' not in request.files:
        print("画像ファイルが見つかりません: リクエストに 'image' 部分がありません") # デバッグ用にログ出力
        return jsonify({"error": "画像ファイルが見つかりません"}), 400
    
    file = request.files['image']

    try:
        # Pillowで画像データを開く
        img = Image.open(file.stream)
        
        # 画像処理と分類を実行
        result = process_image(img)
        
        # UnityからPOSTで画像が送られてきた場合も、その結果を最新とする
        last_classification_result = result 
        print(f"Unityからの画像識別完了。結果を保存しました: {result}") # デバッグ用にログ出力
        
        # 結果をJSONで返す (UnityのImageClassifierClientが直接この結果を受け取る)
        return jsonify(result)

    except Exception as e:
        print(f"画像処理中にエラーが発生しました: {e}") # デバッグ用にエラーログ出力
        return jsonify({"error": str(e)}), 500


# CORSを有効にする。全てのオリジンからのリクエストを許可する設定
CORS(app) 
# --- 追加ここまで ---


# Unityが最新の識別結果をGETリクエストで取得するためのエンドポイント
@app.route("/get_last_result", methods=["GET"])
def get_last_result():
    # 保存されている最新の結果を返す
    print(f"最新結果のリクエストを受信。返却: {last_classification_result}") # デバッグ用にログ出力
    return jsonify(last_classification_result)


if __name__ == "__main__":
    # サーバー起動時に一度だけ固定画像を識別し、結果をセットする
    # これは、Unityがポーリングする際に最初の結果がすぐに得られるようにするためです。
    process_fixed_image_periodically()

    # Flaskサーバーを起動
    # host='0.0.0.0' で、ローカルネットワーク内の他のデバイス(Unity実行PC)からアクセス可能に
    # debug=True は開発中は便利ですが、本番環境では False にしてください。
    print("サーバーを起動します。")
    app.run(host='0.0.0.0', port=5000, debug=True)
