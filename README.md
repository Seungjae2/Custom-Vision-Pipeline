# Azure-Custom-Vision-YouTube-Pipeline: 자동화된 썸네일 분석 및 모델 개선 파이프라인

매일 유튜브 인기 동영상의 썸네일을 자동으로 수집하고, **Azure Custom Vision**을 활용하여 썸네일 안의 객체(인물, 로고 등)를 탐지하는 완전 자동화 머신러닝 파이프라인입니다.

특히, 안정적인 예측을 수행하는 **프로덕션 모델(A)**과 새로운 데이터로 모델을 개선하는 **개발 모델(B)**을 분리하여, 서비스 중단 없이 안전하게 모델을 업데이트할 수 있는 견고한 MLOps 아키텍처를 채택하고 있습니다.

## **🚀 핵심 기능**

* **완전 자동화 MLOps 파이프라인**
매일 정해진 시간에 자동으로 데이터를 수집, 예측, 재학습하는 모든 과정을 수행합니다.
* **안정적인 A/B 프로젝트 아키텍처**
    * **예측(Prediction)**: 검증된 **프로젝트 A**의 모델을 사용하여 안정적인 예측 결과를 도출합니다.
    * **학습(Upload \& Train)**: 예측 결과를 **프로젝트 B**에 업로드하여, 새로운 데이터로 모델을 지속적으로 개선하고 관리합니다.
* **지능적인 태그 관리 및 유효성 검사**
    * **예측기 (Predictor)**: 코드에 정의된 태그(`LABEL_INFO`)와 Azure 프로젝트 A의 실제 태그가 100% 일치하는지 **엄격하게 검증**하여, 설정 불일치로 인한 오류를 원천 차단합니다.
    * **업로더 (Uploader)**: 학습용 프로젝트 B에 필요한 태그가 없을 경우, **자동으로 태그를 생성**하여 파이프라인이 멈추지 않도록 합니다.
* **견고한 오류 처리 및 데이터 관리**
    * 파일 이름에 포함된 **이모지 및 특수문자를 완벽하게 제거**하여 파일 시스템 오류를 방지합니다.
    * 일부 이미지 다운로드에 실패하거나 파일이 손상되어도 전체 파이프라인이 중단되지 않고 해당 파일만 건너뜁니다.
    * 이미 업로드된 이미지는 **중복으로 업로드하지 않아** 리소스 낭비를 막습니다.
* **유연한 실행 모드**
    * **스케줄 모드**: 지정된 시간에 맞춰 매일 자동으로 실행됩니다.
    * **즉시 실행 모드**: `--now` 옵션을 통해 필요할 때 언제든지 수동으로 즉시 실행할 수 있습니다.


## **⚙️ 시스템 아키텍처 흐름**

이 파이프라인은 아래와 같은 순서로 동작합니다.

`[1. 데이터 수집: crawler.py]` ➡️ `[2. 객체 탐지: azure_predictor.py (프로젝트 A 사용)]` ➡️ `[3. 모델 재학습: azure_uploader.py (프로젝트 B에 업로드)]`

## **🛠️ 설치 및 초기 설정**

### **1. 사전 준비물**

* Python 3.8 이상
* Git


### **2. 프로젝트 클론 및 환경 설정**

1. **GitHub에서 프로젝트 클론**: 터미널을 열고, 아래 명령어로 프로젝트를 컴퓨터에 복제합니다[memory.1].

```bash
git clone [GitHub 저장소 주소]
cd Youtube-azure-pipeline
```

2. **가상 환경 생성 및 활성화**: 프로젝트 폴더 내에 격리된 파이썬 환경을 만듭니다.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. **필요 라이브러리 설치**: `requirements.txt` 파일을 이용해 모든 라이브러리를 한 번에 설치합니다.

```bash
pip install -r requirements.txt
```


### **3. Azure Custom Vision 설정**

1. Azure Portal에 로그인하여 **두 개의 Custom Vision 프로젝트**를 생성합니다.
    * **프로젝트 A**: 예측에 사용할, 이미 학습이 완료된 안정적인 프로젝트.
    * **프로젝트 B**: 새로운 데이터를 쌓고 모델을 개선할, 비어있거나 개발 중인 프로젝트.
2. **프로젝트 A**의 **`Settings`** 페이지로 이동하여 **프로젝트 ID(Project Id)**를 복사합니다.
3. **프로젝트 B**의 **`Settings`** 페이지로 이동하여 **프로젝트 ID(Project Id)**를 복사합니다.
4. Azure Portal의 **Custom Vision 리소스** 페이지로 이동하여 아래 정보들을 복사합니다.
    * **학습(Training) 리소스**: `Training Key`와 `Training Endpoint`
    * **예측(Prediction) 리소스**: `Prediction Key`, `Prediction Endpoint`, `Prediction Resource ID`

### **4. `.env` 파일 설정 (가장 중요!)**

1. 프로젝트 최상위 폴더에 `.env` 라는 이름의 파일을 생성합니다.
2. 아래 템플릿을 복사하여 붙여넣고, **3단계에서 복사한 실제 값**으로 채워 넣습니다.

```env
# --- 학습(Training) 리소스 정보 ---
AZURE_TRAINING_KEY="your_training_key"
AZURE_TRAINING_ENDPOINT="https://your-training-endpoint.cognitiveservices.azure.com/"

# --- 예측(Prediction) 리소스 정보 ---
AZURE_PREDICTION_KEY="your_prediction_key"
AZURE_PREDICTION_ENDPOINT="https://your-prediction-endpoint.cognitiveservices.azure.com/"

# --- 프로젝트 ID 분리 ---
# [프로젝트 A - 예측용]
AZURE_PREDICTION_PROJECT_ID="project_A_id"

# [프로젝트 B - 업로드/학습용]
AZURE_UPLOADER_PROJECT_ID="project_B_id"

# --- 공통 정보 ---
# 예측 리소스의 ARM ID (게시(publish)에 필요)
AZURE_PREDICTION_RESOURCE_ID="your_prediction_resource_arm_id"
```


## **▶️ 사용 방법**

### **전체 파이프라인 실행**

터미널에서 `main.py`를 실행합니다.

* **스케줄 모드 (매일 03:00 자동 실행)**:

```bash
python main.py
```

* **즉시 실행 모드 (지금 당장 1회 실행)**:

```bash
python main.py --now
```


### **개별 스크립트 실행 (테스트 및 디버깅용)**

각 모듈의 기능을 개별적으로 테스트할 수 있습니다.

* **데이터 수집만 실행**:

```bash
python crawler.py
```

* **예측만 실행** (가장 최근에 수집된 데이터 대상):

```bash
python azure_predictor.py
```

* **업로드/학습만 실행** (가장 최근 데이터 대상):

```bash
python azure_uploader.py
```


## **📂 프로젝트 파일 구조**

```
/youtube_azure_pipeline/
│
├── .env                  # (가장 중요) Azure API 키 등 민감 정보 저장 파일
├── .gitignore            # Git에 올리지 않을 파일/폴더 목록 (예: .env, data/)
├── requirements.txt      # 프로젝트에 필요한 파이썬 라이브러리 목록
│
├── crawler.py            # 1. 유튜브 썸네일 수집기 (이모지, 특수문자 제거 기능 포함)
├── azure_predictor.py    # 2. [프로젝트 A]를 사용한 예측기 (엄격한 태그 유효성 검사 기능 포함)
├── azure_uploader.py     # 3. [프로젝트 B]에 업로드 및 재학습 (태그 자동 생성, 이미지 중복 방지 기능 포함)
└── main.py               # 전체 파이프라인 실행 및 스케줄 관리 (즉시/스케줄 모드)
```

