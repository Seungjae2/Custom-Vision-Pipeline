import time
import os
import re
from datetime import datetime
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def get_high_quality_thumbnail_url(video_id):
    """
    동영상 ID를 기반으로 가능한 최고 화질의 썸네일 URL을 생성하고 유효성을 검사합니다.
    """
    # 1순위: 최대 해상도 썸네일 시도
    maxres_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    try:
        response = requests.head(maxres_url, timeout=5)
        if response.status_code == 200:
            return maxres_url
    except requests.RequestException:
        pass  # 연결 실패 시 다음으로 넘어감

    # 2순위: 720p 고화질 썸네일 시도
    hq720_url = f"https://i.ytimg.com/vi/{video_id}/hq720.jpg"
    try:
        response = requests.head(hq720_url, timeout=5)
        if response.status_code == 200:
            return hq720_url
    except requests.RequestException:
        pass

    # 3순위: 표준 고화질 썸네일 (거의 항상 존재)
    hqdefault_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    return hqdefault_url


def download_and_verify_image(url, path, title):
    """
    주어진 URL에서 이미지를 다운로드하고, 성공적으로 저장되었는지 검증합니다.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        img_data = requests.get(url, headers=headers, timeout=10).content

        with open(path, 'wb') as handler:
            handler.write(img_data)

        # --- ★★★ 디버깅 강화: 다운로드 완료 검증 ★★★ ---
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  [성공] 썸네일 저장 완료: {os.path.basename(path)}")
            return True
        else:
            print(f"  [실패] 썸네일 파일 생성 실패 또는 파일 크기 0: {title}")
            return False

    except Exception as e:
        print(f"  [오류] 썸네일 다운로드 중 오류 발생: {e}")
        return False


def crawl_youtube_trending():
    youtube_trending_url = "https://www.youtube.com/feed/trending?bp=6gQJRkVleHBsb3Jl"

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.join("data", f"youtube_trending_{timestamp_str}")
    image_folder = os.path.join(base_folder, "thumbnails")
    os.makedirs(image_folder, exist_ok=True)
    print(f"폴더 생성: {image_folder}")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=ko_KR")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(options=options)
    print("WebDriver 시작됨")

    try:
        driver.get(youtube_trending_url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-video-renderer")))

        last_count = 0
        same_count_repeat = 0
        for i in range(30):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2.5)
            video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
            if len(video_elements) == last_count:
                same_count_repeat += 1
            else:
                same_count_repeat = 0
            if same_count_repeat >= 2:
                break
            last_count = len(video_elements)

        all_video_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        print(f"\n총 {len(all_video_elements)}개의 동영상 렌더러 발견. 데이터 추출 시작...")

        video_data = []
        for i, video_element in enumerate(all_video_elements):
            try:
                title_element = video_element.find_element(By.CSS_SELECTOR, "a#video-title")
                title = title_element.get_attribute("title")
                link = title_element.get_attribute("href")

                if not link or "watch?v=" not in link:
                    continue

                # --- ★★★ 화질 개선: 동영상 ID 추출 ★★★ ---
                video_id = link.split('watch?v=')[1].split('&')[0]

                print(f"\n[{i + 1}/{len(all_video_elements)}] 처리 중: {title}")

                # 고화질 썸네일 URL 생성
                thumbnail_url = get_high_quality_thumbnail_url(video_id)
                print(f"  고화질 썸네일 URL 확보: {thumbnail_url}")

                safe_title = sanitize_filename(title)[:50]
                image_filename = f"rank_{i + 1:03d}_{safe_title}.jpg"
                image_path = os.path.join(image_folder, image_filename)

                # 다운로드 및 검증
                download_success = download_and_verify_image(thumbnail_url, image_path, title)

                if download_success:
                    video_data.append({
                        "rank": i + 1, "title": title, "link": link, "thumbnail_file": image_filename
                    })
            except Exception as e_loop:
                # print(f"  동영상 [{i+1}] 처리 중 오류, 건너뜁니다: {e_loop}")
                continue

        if video_data:
            df = pd.DataFrame(video_data)
            csv_path = os.path.join(base_folder, f"youtube_trending_rankings_{timestamp_str}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n데이터 저장 완료: {csv_path} ({len(video_data)}개 수집)")
            return image_folder
        else:
            print("\n수집된 유효한 데이터가 없습니다.")
            return None

    except Exception as e:
        print(f"크롤링 중 심각한 오류 발생: {e}")
        return None
    finally:
        driver.quit()
        print("WebDriver 종료됨")


if __name__ == "__main__":
    print("크롤러 모듈을 단독으로 테스트 실행합니다.")
    result_folder = crawl_youtube_trending()
    if result_folder:
        print(f"\n[테스트 성공] 썸네일이 저장된 경로: {result_folder}")
    else:
        print("\n[테스트 실패] 크롤링에 실패했거나 수집된 데이터가 없습니다.")
