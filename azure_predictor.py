# azure_predictor.py

import os
import json
import requests
from PIL import Image
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 설정 (학습/예측 리소스 정보 모두 사용) ---
PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
PREDICTION_ENDPOINT = os.getenv("AZURE_PREDICTION_ENDPOINT")
TRAINING_KEY = os.getenv("AZURE_TRAINING_KEY")
TRAINING_ENDPOINT = os.getenv("AZURE_TRAINING_ENDPOINT")
# 예측을 수행할 프로젝트 A의 ID를 불러옵니다.
PROJECT_ID = os.getenv("AZURE_PREDICTION_PROJECT_ID")

# --- 이 부분이 이제 '진실의 원천(Single Source of Truth)'이 됩니다 ---
LABEL_INFO = {
    "브랜드/로고": {"id": 1, "threshold": 0.8},
    "인물": {"id": 2, "threshold": 0.8},
    "캐릭터": {"id": 3, "threshold": 0.7},
    "텍스트": {"id": 4, "threshold": 0.8},
}


# --- 신규 함수: Azure 프로젝트의 태그와 코드의 LABEL_INFO를 비교 검증 ---
def validate_azure_tags(project_id, label_info_from_code):
    """
    Azure Custom Vision 프로젝트의 실제 태그 목록과 코드에 정의된 LABEL_INFO가
    정확히 일치하는지 검증합니다.
    """
    print(f"\n🔄 예측 프로젝트(ID: {project_id})의 태그 유효성을 검사합니다...")

    get_tags_url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{project_id}/tags"
    headers = {"Training-Key": TRAINING_KEY}

    try:
        res = requests.get(get_tags_url, headers=headers)
        res.raise_for_status()
        api_tags = res.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Azure에서 태그 목록 조회 실패: {e}");
        return False

    # 비교를 위해 Set 자료형 사용 (순서와 상관없이 내용만 비교)
    azure_tag_names = set(tag['name'] for tag in api_tags)
    code_tag_names = set(label_info_from_code.keys())

    if azure_tag_names == code_tag_names:
        print("  - ✅ 태그 일치. 유효성 검사를 통과했습니다.")
        return True
    else:
        # 불일치 시, 어떤 부분이 다른지 상세히 알려줌
        print("❌ 태그 불일치! 예측을 중단합니다.")
        missing_in_azure = code_tag_names - azure_tag_names
        extra_in_azure = azure_tag_names - code_tag_names
        if missing_in_azure:
            print(f"  - ❗️ Azure 프로젝트에 누락된 태그: {missing_in_azure}")
        if extra_in_azure:
            print(f"  - ❗️ 코드에 정의되지 않은 태그가 Azure 프로젝트에 존재: {extra_in_azure}")
        return False


def get_latest_published_iteration_url(project_id):
    # ... (이전과 동일, 변경 없음)
    iterations_url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{project_id}/iterations"
    headers = {"Training-Key": TRAINING_KEY}
    try:
        res = requests.get(iterations_url, headers=headers);
        res.raise_for_status()
        iterations = [it for it in res.json() if it.get("publishName")]
        if not iterations: raise RuntimeError("게시된 Iteration이 없습니다.")
        latest_iteration = sorted(iterations, key=lambda it: it["lastModified"], reverse=True)[0]
        print(f"✅ 예측에 사용될 Iteration: '{latest_iteration['publishName']}'")
        return f"{PREDICTION_ENDPOINT}customvision/v3.0/Prediction/{project_id}/detect/iterations/{latest_iteration['publishName']}/image"
    except Exception as e:
        print(f"❌ Iteration 정보 조회 실패: {e}"); return None


def predict_image(image_path, prediction_url):
    # ... (이전과 동일, 변경 없음)
    with open(image_path, "rb") as f:
        response = requests.post(prediction_url,
                                 headers={"Prediction-Key": PREDICTION_KEY, "Content-Type": "application/octet-stream"},
                                 data=f)
    response.raise_for_status()
    return response.json()


def convert_to_coco(image_folder, prediction_url, label_info):
    # ... (이전과 동일, 변경 없음)
    coco = {"images": [], "annotations": [], "categories": [{"id": v["id"], "name": k} for k, v in label_info.items()]}
    annotation_id_counter = 1
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for image_id_counter, file_name in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, file_name)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            coco["images"].append({"id": image_id_counter, "file_name": file_name, "width": width, "height": height})
            prediction_result = predict_image(image_path, prediction_url)
            current_image_annotations = []
            for pred in prediction_result.get("predictions", []):
                tag_name = pred.get("tagName");
                probability = pred.get("probability")
                if tag_name in label_info and probability >= label_info[tag_name]["threshold"]:
                    bbox = pred["boundingBox"]
                    x, y, w, h = (bbox["left"] * width, bbox["top"] * height, bbox["width"] * width,
                                  bbox["height"] * height)
                    new_annotation = {"id": annotation_id_counter, "image_id": image_id_counter,
                                      "category_id": label_info[tag_name]["id"], "bbox": [x, y, w, h], "area": w * h,
                                      "iscrowd": 0, "score": probability}
                    current_image_annotations.append(new_annotation);
                    annotation_id_counter += 1
            coco["annotations"].extend(current_image_annotations)
            print(f"  - 처리 완료: {file_name} (유효 예측 {len(current_image_annotations)}개 추가)")
        except Exception as e:
            print(f"  - 예측 실패: {file_name}, 오류: {e}"); continue
    return coco


# --- 이 아래 run_prediction 함수가 수정되었습니다 ---
def run_prediction(image_folder, output_coco_path):
    if not all([PREDICTION_KEY, PREDICTION_ENDPOINT, TRAINING_KEY, TRAINING_ENDPOINT, PROJECT_ID]):
        print("❌ .env 파일에 Azure 설정값이 모두 지정되지 않았습니다.");
        return False

    # 1. Azure 프로젝트의 태그와 코드의 LABEL_INFO가 일치하는지 검증합니다.
    if not validate_azure_tags(PROJECT_ID, LABEL_INFO):
        print("\n❌ 태그 설정 불일치로 인해 파이프라인을 중단합니다.")
        return False

    print(f"\n▶️ Azure Predictor 시작 (대상 폴더: {image_folder})")
    prediction_url = get_latest_published_iteration_url(PROJECT_ID)
    if not prediction_url: return False

    coco_data = convert_to_coco(image_folder, prediction_url, LABEL_INFO)

    os.makedirs(os.path.dirname(output_coco_path), exist_ok=True)
    with open(output_coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 예측 완료. 결과 저장: {output_coco_path}")
    print(f"  (이미지: {len(coco_data['images'])}개, 탐지된 객체: {len(coco_data['annotations'])}개)")
    return True


if __name__ == "__main__":
    # ... (이전과 동일, 변경 없음)
    print("--- 예측 모듈 단독 테스트 실행 ---");
    print("크롤링으로 생성된 최신 데이터 폴더를 자동으로 탐색합니다...")
    base_data_dir = "data";
    latest_crawled_folder = None
    if os.path.exists(base_data_dir):
        all_subdirs = [d for d in os.listdir(base_data_dir) if
                       os.path.isdir(os.path.join(base_data_dir, d)) and d.startswith("youtube_trending_")]
        if all_subdirs: latest_crawled_folder = max(all_subdirs)

    if latest_crawled_folder:
        image_folder_path = os.path.join(base_data_dir, latest_crawled_folder, "thumbnails")
        print(f"✅ 최신 폴더 발견: {image_folder_path}")
        output_path = os.path.join(base_data_dir, latest_crawled_folder, "predictions.json")
        if os.path.exists(output_path): print(f"🗑️ 기존 '{output_path}' 파일을 삭제합니다."); os.remove(output_path)
        if os.path.isdir(image_folder_path):
            run_prediction(image_folder_path, output_path)
        else:
            print(f"❌ 오류: 이미지 폴더 '{image_folder_path}'를 찾을 수 없습니다.")
    else:
        print("❌ 크롤링 데이터 폴더를 찾을 수 없습니다. 먼저 `crawler.py`를 실행하여 데이터를 수집해주세요.")
