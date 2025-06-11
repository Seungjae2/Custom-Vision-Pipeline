# azure_uploader.py

import os
import json
import base64
import requests
import time
from urllib.parse import quote
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 설정 (업로드/학습용 프로젝트 B의 정보 사용) ---
TRAINING_KEY = os.getenv("AZURE_TRAINING_KEY")
TRAINING_ENDPOINT = os.getenv("AZURE_TRAINING_ENDPOINT")
PROJECT_ID = os.getenv("AZURE_UPLOADER_PROJECT_ID")
PREDICTION_RESOURCE_ID = os.getenv("AZURE_PREDICTION_RESOURCE_ID")
USE_ADVANCED_TRAINING = True

TRAIN_HEADERS = {"Training-Key": TRAINING_KEY, "Content-Type": "application/json"}


# --- 신규 함수: Azure 프로젝트의 기존 이미지 목록 조회 ---
def get_existing_images_from_azure():
    """Azure Custom Vision 프로젝트에 이미 업로드된 모든 이미지의 파일 이름을 가져옵니다."""
    existing_images = set()
    page_num = 0

    print("\n☁️ Azure에서 기존 이미지 목록을 확인합니다...")

    while True:
        # get_tagged_images API는 페이징 처리가 필요 (한 번에 최대 256개씩 조회)
        url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/images?take=256&skip={page_num * 256}"
        try:
            res = requests.get(url, headers=TRAIN_HEADERS)
            res.raise_for_status()
            images_on_page = res.json()

            if not images_on_page:
                break  # 더 이상 가져올 이미지가 없으면 루프 종료

            for image in images_on_page:
                # Azure는 파일 이름을 'name' 필드에 저장
                if 'name' in image:
                    existing_images.add(image['name'])
            page_num += 1

        except requests.exceptions.RequestException as e:
            print(f"❌ 기존 이미지 목록 조회 실패: {e}")
            return None  # 오류 발생 시 None 반환

    print(f"  - ✅ {len(existing_images)}개의 기존 이미지를 확인했습니다.")
    return existing_images


def sync_and_get_tags(required_tag_names):
    """
    Azure 프로젝트의 태그를 동기화하고 최종 태그 맵을 반환합니다.
    """
    print("\n🔄 Azure 프로젝트와 태그 동기화를 시작합니다...")
    get_tags_url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/tags"
    try:
        res = requests.get(get_tags_url, headers=TRAIN_HEADERS)
        res.raise_for_status()
        tag_map = {tag['name']: tag['id'] for tag in res.json()}
        print(f"  - 현재 프로젝트에 {len(tag_map)}개의 태그가 있습니다: {list(tag_map.keys())}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 기존 태그 목록 조회 실패: {e}")
        return None

    for tag_name in required_tag_names:
        if tag_name not in tag_map:
            print(f"  - 필요한 태그 '{tag_name}'을(를) 새로 생성합니다...")
            create_tag_url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/tags?name={quote(tag_name)}"
            try:
                res = requests.post(create_tag_url, headers={"Training-Key": TRAINING_KEY})
                res.raise_for_status()
                new_tag_info = res.json()
                tag_map[new_tag_info['name']] = new_tag_info['id']
                print(f"  ✅ 새로운 태그 생성 성공: {new_tag_info['name']}")
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 태그 '{tag_name}' 생성 실패: {e}")
                continue

    print("  - ✅ 태그 동기화 완료.")
    return tag_map


def get_next_iteration_name():
    # ... (이전과 동일, 변경 없음)
    url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/iterations"
    try:
        res = requests.get(url, headers=TRAIN_HEADERS);
        res.raise_for_status();
        iterations = res.json()
    except requests.exceptions.RequestException:
        return "Iteration-1"
    max_num = 0
    for it in iterations:
        if "Iteration" in it["name"]:
            try:
                num_part = it["name"].replace("Iteration", "").strip().replace("-", "");
                num = int(num_part);
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
    return f"Iteration-{max_num + 1}"


def convert_coco_to_azure_format(coco_data, tag_map):
    # ... (이전과 동일, 변경 없음)
    uploads = {};
    images = {img["id"]: img for img in coco_data["images"]};
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    for ann in coco_data["annotations"]:
        image_info = images.get(ann["image_id"]);
        file_name = image_info["file_name"] if image_info else None
        category_name = categories.get(ann["category_id"])
        if file_name and category_name and category_name in tag_map:
            x, y, w, h = ann["bbox"];
            width, height = image_info["width"], image_info["height"]
            region = {"tagId": tag_map[category_name], "left": x / width, "top": y / height, "width": w / width,
                      "height": h / height}
            if file_name not in uploads: uploads[file_name] = []
            uploads[file_name].append(region)
    return uploads


def upload_images_to_azure(image_folder, uploads):
    # ... (이전과 동일, 변경 없음)
    batch, total_sent = [], 0;
    total_images = len(uploads)
    for fname, regions in list(uploads.items()):
        fpath = os.path.join(image_folder, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, "rb") as f:
            b64_content = base64.b64encode(f.read()).decode()
        batch.append({"name": fname, "contents": b64_content, "regions": regions})
        if len(batch) >= 64: total_sent = send_batch(batch, total_sent, total_images); batch = []
    if batch: send_batch(batch, total_sent, total_images)


def send_batch(batch, sent_count, total_count):
    # ... (이전과 동일, 변경 없음)
    url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/images/files"
    print(f"📤 업로드 중... [{sent_count + 1}–{sent_count + len(batch)} / {total_count}]")
    try:
        res = requests.post(url, headers=TRAIN_HEADERS, json={"images": batch}, timeout=180);
        res.raise_for_status();
        print("  - 배치 업로드 성공.")
    except requests.exceptions.RequestException as e:
        print(f"  - ❌ 배치 업로드 실패: {e}")
    return sent_count + len(batch)


def train_new_iteration(iteration_name):
    # ... (이전과 동일, 변경 없음)
    url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/train?advancedTraining={'true' if USE_ADVANCED_TRAINING else 'false'}"
    res = requests.post(url, headers=TRAIN_HEADERS)
    if res.ok:
        print(f"🧠 학습 요청 성공: '{iteration_name}'"); return res.json()
    else:
        print(f"❌ 학습 요청 실패 ({res.status_code}): {res.text}"); return None


def wait_for_training_completion(iteration_id, timeout=3600, interval=30):
    # ... (이전과 동일, 변경 없음)
    url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/iterations/{iteration_id}";
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            res = requests.get(url, headers=TRAIN_HEADERS);
            res.raise_for_status();
            iteration_data = res.json();
            status = iteration_data["status"]
            print(f"⏳ 학습 상태 확인: {status} (경과 시간: {int(time.time() - start_time)}초)")
            if status == "Completed": return True
            if status in ["Failed", "Canceled"]: return False
            time.sleep(interval)
        except requests.exceptions.RequestException as e:
            print(f"  - 학습 상태 확인 중 오류 발생: {e}"); time.sleep(interval)
    print("⏰ 학습 대기 시간 초과");
    return False


def publish_iteration(iteration_id, iteration_name):
    # ... (이전과 동일, 변경 없음)
    encoded_iteration_name = quote(iteration_name)
    url = f"{TRAINING_ENDPOINT}customvision/v3.3/training/projects/{PROJECT_ID}/iterations/{iteration_id}/publish?publishName={encoded_iteration_name}&predictionId={PREDICTION_RESOURCE_ID}"
    res = requests.post(url, headers=TRAIN_HEADERS)
    if res.ok:
        print(f"🚀 게시 성공: '{iteration_name}'")
    else:
        print(f"❌ 게시 실패 ({res.status_code}): {res.text}")


# --- 이 아래 run_uploader 함수가 수정되었습니다 ---
def run_uploader(image_folder, coco_file_path):
    """COCO 파일과 이미지 폴더를 기반으로 Azure 업로드 및 학습 파이프라인을 실행합니다."""
    if not all([TRAINING_KEY, TRAINING_ENDPOINT, PROJECT_ID, PREDICTION_RESOURCE_ID]):
        print("❌ .env 파일에 Azure 설정값이 모두 지정되지 않았습니다.");
        return False

    print(f"\n▶️ Azure Uploader 시작 (대상 파일: {coco_file_path})")
    if not os.path.exists(coco_file_path):
        print(f"❌ COCO 파일({coco_file_path})을 찾을 수 없습니다.");
        return False

    with open(coco_file_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 1. Azure에서 기존 이미지 목록을 가져옵니다.
    existing_images_on_azure = get_existing_images_from_azure()
    if existing_images_on_azure is None:
        print("❌ Azure에서 이미지 목록을 가져오지 못해 업로드를 중단합니다.");
        return False

    # 2. 로컬 JSON 파일에 있는 이미지 목록과 비교하여, 업로드할 새로운 이미지 목록을 만듭니다.
    all_local_images = {img['file_name'] for img in coco_data.get('images', [])}
    new_images_to_upload = all_local_images - existing_images_on_azure

    if not new_images_to_upload:
        print("\n✅ 새로운 이미지가 없습니다. 업로드 및 학습을 건너뜁니다.")
        return True

    print(f"\n🆕 총 {len(all_local_images)}개 이미지 중 {len(new_images_to_upload)}개의 새로운 이미지를 업로드합니다.")

    # 3. 새로운 이미지만 포함하도록 coco_data 필터링
    filtered_coco = {"images": [], "annotations": [], "categories": coco_data["categories"]}
    new_image_ids = set()
    for img in coco_data["images"]:
        if img["file_name"] in new_images_to_upload:
            filtered_coco["images"].append(img)
            new_image_ids.add(img["id"])

    for ann in coco_data.get("annotations", []):
        if ann["image_id"] in new_image_ids:
            filtered_coco["annotations"].append(ann)

    if not filtered_coco.get("annotations"):
        print("⚠️ 새로운 이미지에 대한 주석(annotation) 데이터가 없어 업로드를 건너뜁니다.");
        return True

    required_tag_names = [cat['name'] for cat in filtered_coco.get('categories', [])]
    tag_map = sync_and_get_tags(required_tag_names)
    if tag_map is None:
        print("❌ 태그 맵을 가져오지 못해 업로드를 중단합니다.");
        return False

    uploads = convert_coco_to_azure_format(filtered_coco, tag_map)
    if not uploads:
        print("⚠️ 업로드할 유효한 이미지가 없습니다.");
        return True

    upload_images_to_azure(image_folder, uploads)

    iteration_name = get_next_iteration_name()
    iteration_info = train_new_iteration(iteration_name)

    if iteration_info:
        iteration_id = iteration_info["id"]
        if wait_for_training_completion(iteration_id):
            publish_iteration(iteration_id, iteration_name)
        else:
            print("⚠️ 학습이 완료되지 않아 게시를 생략합니다.")
    return True


if __name__ == "__main__":
    # ... (이전과 동일, 변경 없음)
    print("--- 업로더 모듈 단독 테스트 실행 ---")
    print("크롤링으로 생성된 최신 데이터 폴더와 JSON 파일을 자동으로 탐색합니다...")
    base_data_dir = 'data';
    latest_crawled_folder = None
    if os.path.exists(base_data_dir):
        all_subdirs = [d for d in os.listdir(base_data_dir) if
                       os.path.isdir(os.path.join(base_data_dir, d)) and d.startswith('youtube_trending_')]
        if all_subdirs: latest_crawled_folder = max(all_subdirs)

    if latest_crawled_folder:
        image_folder_path = os.path.join(base_data_dir, latest_crawled_folder, 'thumbnails')
        json_file_path = os.path.join(base_data_dir, latest_crawled_folder, 'predictions.json')
        print(f"✅ 최신 이미지 폴더 발견: {image_folder_path}");
        print(f"✅ 최신 JSON 파일 발견: {json_file_path}")
        if os.path.isdir(image_folder_path) and os.path.isfile(json_file_path):
            run_uploader(image_folder_path, json_file_path)
        else:
            if not os.path.isdir(image_folder_path): print(f"❌ 오류: 이미지 폴더를 찾을 수 없습니다: {image_folder_path}")
            if not os.path.isfile(json_file_path): print(f"❌ 오류: '{json_file_path}' 파일을 찾을 수 없습니다.")
    else:
        print("❌ 크롤링 데이터 폴더를 찾을 수 없습니다. 먼저 `crawler.py`를 실행하여 데이터를 수집해주세요.")
