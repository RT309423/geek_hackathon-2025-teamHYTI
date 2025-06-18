# classify_local.py

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import sys # コマンドライン引数を扱うために追加

def classify_scene(image_path):
    """
    画像ファイルのパスを受け取り、プロジェクト用のカテゴリ名を返す関数
    """
    # 1. 事前学習済みモデルをロードする
    arch = 'resnet18'
    model_file = r'C:\Users\honob\VR_Walk_Project\geek_hackathon-2025-teamHYTI\CNN\resnet18_places365.pth.tar'
    # GPUがない環境でも動作するように map_location を設定
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
    file_name = r'C:\Users\honob\VR_Walk_Project\geek_hackathon-2025-teamHYTI\CNN\categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        # Fallback for systems where file is not writable
        file_name = os.path.join(os.path.dirname(__file__), file_name)

    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])

    # 4. 【重要】365カテゴリを自分たちのカテゴリに変換するマッピング辞書
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
        img = Image.open(image_path)
        input_img = V(centre_crop(img).unsqueeze(0))
    except FileNotFoundError:
        print(f"エラー: ファイル '{image_path}' が見つかりません。")
        return
    except Exception as e:
        print(f"エラー: 画像を読み込めません。({e})")
        return

    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    predicted_by_model = classes[idx[0]]
    final_category = category_mapping.get(predicted_by_model, "その他")

    print("-" * 30)
    print(f"入力ファイル: {image_path}")
    print(f"AIの予測結果: '{predicted_by_model}' (確率: {probs[0]:.3f})")
    print(f"最終的な分類: '{final_category}'")
    print("-" * 30)

image_file_to_test = r'C:\Users\honob\VR_Walk_Project\geek_hackathon-2025-teamHYTI\CNN\my_photo.jpg' 
classify_scene(image_file_to_test)

"""""
# --- メインの実行部分 ---
if __name__ == '__main__':
    # コマンドラインからファイル名を受け取る
    if len(sys.argv) != 2:
        print("使い方: python classify_local.py [画像ファイル名]")
    else:
        image_file_to_test = sys.argv[1]
        classify_scene(image_file_to_test)
"""