# classify_local.py

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import sys
import numpy as np # --- 追加 ---

# --- 追加ここから ---
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

    # --- 判定ロジック（このしきい値は調整可能） ---
    BRIGHTNESS_NIGHT = 60   # この明るさより暗ければ「夜」
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
# --- 追加ここまで ---


def classify_scene(image_path):
    """
    画像ファイルのパスを受け取り、プロジェクト用のカテゴリ名と時間帯を返す関数
    """
    # 1. 事前学習済みモデルをロードする
    arch = 'resnet18'
    model_file = r'C:\Users\保延荘志\geek2025\geek_hackathon-2025-teamHYTI-1\CNN\resnet18_places365.pth.tar'
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 画像をAIが読み取れる形式に変換する設定
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 365カテゴリのラベル情報をロード
    file_name = r'C:\Users\保延荘志\geek2025\geek_hackathon-2025-teamHYTI-1\CNN\categories_places365.txt'
    # --- 以下のパスの部分は、元のままでも動くなら変更不要です ---
    if not os.path.exists(file_name):
        # スクリプトと同じディレクトリにあると仮定してパスを再設定
        file_name = os.path.join(os.path.dirname(__file__), 'categories_places365.txt')

    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])

    # 4. カテゴリのマッピング辞書
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

    # 5. 画像を読み込んで分類を実行
    try:
        img = Image.open(image_path) # Pillowで画像を開く
        input_img = V(centre_crop(img).unsqueeze(0))
    except FileNotFoundError:
        print(f"エラー: ファイル '{image_path}' が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: 画像を読み込めません。({e})")
        return

    # --- 変更ここから ---
    # 場所の分類
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    predicted_by_model = classes[idx[0]]
    final_category = category_mapping.get(predicted_by_model, "その他")

    # 時間帯の分類
    time_of_day = analyze_time_of_day(img) # ここで新しい関数を呼び出す

    # 結果をまとめて出力
    print("-" * 30)
    print(f"入力ファイル: {image_path}")
    print(f"AIの予測結果 (場所): '{predicted_by_model}' (確率: {probs[0]:.3f})")
    print(f"最終的な分類 (場所): '{final_category}'")
    print(f"最終的な分類 (時間帯): '{time_of_day}'") # 時間帯の結果も出力
    print("-" * 30)
    
    # Unity側で使いやすいように辞書形式で結果を返すこともできます
    return {"place": final_category, "time": time_of_day}
    # --- 変更ここまで ---

# --- 以下の実行部分は変更なし ---
image_file_to_test = r'C:\Users\保延荘志\geek2025\geek_hackathon-2025-teamHYTI-1\CNN\my_photo.jpg' 
classify_scene(image_file_to_test)