# main.py

import schedule
import time
import threading
from datetime import datetime
import os
import sys  # 커맨드 라인 인자를 읽기 위해 sys 모듈을 임포트합니다.

# 각 모듈에서 실행 함수 임포트
import crawler
import azure_predictor
import azure_uploader


def run_pipeline():
    """
    크롤링, 예측, 업로드/학습으로 이어지는 전체 파이프라인을 실행합니다.
    """
    print(f"\n{'=' * 50}")
    print(f"🚀 파이프라인 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}")

    # 1. 크롤링 수행
    print("\n[1/3] 유튜브 썸네일 크롤링 시작...")
    image_folder = crawler.crawl_youtube_trending()

    if not image_folder:
        print("❌ 크롤링 실패. 파이프라인을 중단합니다.")
        return
    print(f"✅ 크롤링 성공. 이미지 저장 경로: {image_folder}")

    # 2. 예측 수행
    base_data_folder = os.path.dirname(image_folder)
    prediction_output_path = os.path.join(base_data_folder, "predictions.json")

    # 예측 실행 전, 이전 예측 파일 삭제
    if os.path.exists(prediction_output_path):
        print(f"🗑️ 기존 '{prediction_output_path}' 파일을 삭제합니다.")
        os.remove(prediction_output_path)

    print("\n[2/3] Azure 객체 탐지 예측 시작...")
    prediction_success = azure_predictor.run_prediction(image_folder, prediction_output_path)

    if not prediction_success:
        print("❌ 예측 실패. 파이프라인을 중단합니다.")
        return
    print("✅ 예측 성공.")

    # 3. 업로드 및 학습 수행
    print("\n[3/3] Azure 업로드 및 학습 시작...")
    uploader_success = azure_uploader.run_uploader(image_folder, prediction_output_path)

    if not uploader_success:
        print("❌ 업로드 및 학습 실패.")
    else:
        print("✅ 업로드 및 학습 성공.")

    print(f"\n{'=' * 50}")
    print(f"🎉 파이프라인 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}")


def run_threaded(job_func):
    """스레드를 사용하여 작업을 실행합니다. (긴 작업이 스케줄러를 막지 않도록 함)[2]"""
    job_thread = threading.Thread(target=job_func)
    job_thread.start()


# --- 이 아래 부분이 요청에 따라 수정되었습니다 ---
if __name__ == "__main__":

    # 커맨드 라인에 '--now' 인자가 있는지 확인
    if len(sys.argv) > 1 and sys.argv[1] == '--now':
        # 즉시 실행 모드
        print("▶️ 즉시 실행 모드로 파이프라인을 1회 실행합니다.")
        run_pipeline()  # 즉시 실행 시에는 스레드 없이 바로 실행하여 로그를 순서대로 확인
    else:
        # 스케줄 모드 (기본)
        print("🗓️ 스케줄 모드로 시작합니다. 매일 03:00에 작업이 자동으로 실행됩니다.")
        print("   지금 바로 1회 실행하려면 'python main.py --now' 명령어를 사용하세요.")

        # 매일 새벽 3시에 파이프라인 실행 예약
        schedule.every().day.at("03:00").do(run_threaded, run_pipeline)

        # 스케줄러 루프 실행
        while True:
            schedule.run_pending()
            time.sleep(1)
