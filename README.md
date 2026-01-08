# Road Sign Detection with YOLOv8 (Kaggle Road Sign Dataset)

이 프로젝트는 Kaggle의 **Road Sign Detection** 데이터셋을 이용하여
`YOLOv8s` 모델로 **교통 표지판(신호등, 정지, 제한속도, 횡단보도)**을 탐지하는 실습 예제입니다.

Google Colab 환경(Tesla T4 GPU)에서 실행한 기준으로 정리되어 있습니다.

---

## 1. 환경 및 요구 사항

* 실행 환경: Google Colab
* 주요 라이브러리

  * Python 3.12
  * PyTorch 2.9.0 + CUDA 12.6
  * ultralytics 8.3.248 (YOLOv8)
  * OpenCV, matplotlib, etc.

Colab에서 기본 제공되는 환경을 사용하며, 추가로 `ultralytics`만 설치합니다.

```bash
!pip install ultralytics
```

---

## 2. 데이터셋

* 사용 데이터셋:
  [Kaggle – Road Sign Detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)

* Kaggle API를 이용해 다운로드합니다. 먼저 `kaggle.json` 업로드 후 설정합니다.

```python
from google.colab import files
files.upload()  # kaggle.json 업로드

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d andrewmvd/road-sign-detection
!unzip -q road-sign-detection.zip
```

---

## 3. 디렉토리 구조 준비

YOLOv8 학습에 맞는 커스텀 디렉토리 구조를 생성합니다.

```python
from pathlib import Path

root = Path('./sign_datasets')
root.mkdir(parents=True, exist_ok=True)

for path1 in ('images', 'labels'):
    for path2 in ('train', 'test', 'val'):
        new_path = root / 'SIGN' / path1 / path2
        new_path.mkdir(parents=True, exist_ok=True)
```

최종적으로 구조는 다음과 같습니다.

```text
sign_datasets/
└── SIGN/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## 4. VOC → YOLO 포맷 변환 (convert2Yolo 사용)

원본 데이터셋 어노테이션은 VOC 형식(XML)이므로, `convert2Yolo` 스크립트를 이용하여 YOLO 포맷으로 변환합니다.

### 4.1 convert2Yolo 클론 및 세팅

```bash
%cd /content/sign_datasets
!git clone https://github.com/ssaru/convert2Yolo.git
%cd convert2Yolo/
```

> `requirements.txt`에 있는 `matplotlib==2.2.2` 설치 시, 최신 Colab 환경에서는 빌드 오류가 발생할 수 있습니다.
> 실제 변환 실행(`example.py`)에는 큰 문제가 없으므로, `pip install -r requirements.txt`에서 에러가 나더라도 무시하고 진행할 수 있습니다.

### 4.2 클래스 파일 생성

```python
path = "/content/sign_datasets/convert2Yolo/sign.names"

content = """trafficlight
stop
speedlimit
crosswalk
"""

with open(path, "w") as f:
    f.write(content)
```

### 4.3 VOC → YOLO 변환 실행

```bash
!python example.py \
    --datasets VOC \
    --img_path /content/images \
    --label /content/annotations \
    --convert_output_path /content/sign_datasets/SIGN/labels/train \
    --img_type '.png' \
    --manifest_path /content/sign_datasets \
    --cls_list_file ./sign.names
```

* `/content/images`: 원본 이미지 폴더
* `/content/annotations`: VOC XML 라벨 폴더
* 변환된 YOLO 라벨은 `SIGN/labels/train`에 저장됩니다.

---

## 5. 이미지 이동 (train 이미지 정리)

변환에 사용된 이미지를 YOLO 학습용 `images/train` 폴더로 이동합니다.

```python
import shutil
import os

src = "/content/images"
dst = "/content/sign_datasets/SIGN/images/train"

files = os.listdir(src)

for f in files:
    shutil.move(os.path.join(src, f), os.path.join(dst, f))
```

---

## 6. YOLOv8 설정 파일(custom_voc.yaml) 작성

YOLOv8에서 사용할 데이터 설정 파일을 작성합니다.

```python
path = "/content/sign_datasets/custom_voc.yaml"

content = """path: /content/sign_datasets/SIGN
train:
  - images/train
val:
  - images/val
test:
  - images/test

nc: 4
names: ['Traffic Light', 'Stop', 'Speedlimit', 'Crosswalk']
"""

with open(path, "w") as f:
    f.write(content)
```

* `nc`: 클래스 개수 (4개)
* `names`: 클래스 이름 리스트

---

## 7. Train / Val 데이터 나누기

일부 이미지를 검증(Validation) 용도로 분리합니다.
예시에서는 `road100` ~ `road150` 이름의 파일을 train → val로 이동했습니다.

```python
import os
import shutil

label_train = "/content/sign_datasets/SIGN/labels/train"
label_val = "/content/sign_datasets/SIGN/labels/val"

img_train = "/content/sign_datasets/SIGN/images/train"
img_val = "/content/sign_datasets/SIGN/images/val"

for i in range(100, 151):
    base = f"road{i}"

    # 라벨 이동
    label_src = os.path.join(label_train, base + ".txt")
    label_dst = os.path.join(label_val, base + ".txt")

    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)
    else:
        print(f"라벨 없음: {label_src}")

    # 이미지 이동
    img_src = os.path.join(img_train, base + ".png")
    img_dst = os.path.join(img_val, base + ".png")

    if os.path.exists(img_src):
        shutil.move(img_src, img_dst)
    else:
        print(f"이미지 없음: {img_src}")
```

---

## 8. YOLOv8s 학습

### 8.1 모델 로드

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # COCO로 pretrain된 YOLOv8s
```

### 8.2 학습 실행

```python
results = model.train(
    data='/content/sign_datasets/custom_voc.yaml',
    epochs=10,
    batch=32,
    imgsz=640,
    device=0,
    workers=2,
    name='custom_s'
)
```

* `epochs=10`: 에폭 수
* `batch=32`: 배치 크기
* `imgsz=640`: 입력 이미지 크기
* `device=0`: GPU 사용 (CPU는 `"cpu"`)

학습 로그 예시(요약):

* 전체 검증 성능(mAP@0.5): **0.785**
* mAP@0.5:0.95: **0.639**

클래스별 mAP 예시:

* Traffic Light: mAP@0.5 ≈ 0.41
* Speedlimit: mAP@0.5 ≈ 0.948
* Crosswalk: mAP@0.5 ≈ 0.995, mAP@0.5:0.95 ≈ 0.835

`Stop` 클래스는 검증셋에 표본이 충분하지 않은 경우, 로그에 별도 항목으로 표시되지 않을 수 있습니다.

학습이 끝나면 가중치는 아래 경로에 저장됩니다.

```text
/content/runs/detect/custom_s/weights/
  ├── last.pt
  └── best.pt
```

---

## 9. 학습된 모델로 추론(테스트)

### 9.1 best.pt 로드

```python
from ultralytics import YOLO

model = YOLO("/content/runs/detect/custom_s/weights/best.pt")
```

### 9.2 테스트 이미지 폴더에서 탐지 수행

```python
results = model.predict(
    source="/content/sign_datasets/SIGN/images/test",  # 테스트 이미지 폴더
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,       # 결과 이미지 저장
    save_txt=True,   # 라벨 txt 저장 (YOLO 포맷)
    save_conf=True   # confidence도 함께 저장
)

print(results)
```

* 탐지 결과 이미지는 기본적으로 `/content/runs/detect/predict` 폴더에 저장됩니다.
* `labels` 폴더에는 각 이미지별 탐지 결과가 YOLO 포맷(.txt)으로 저장됩니다.

추론 로그 예시:

* 특정 이미지에서 `Stop` 1개 탐지
* 특정 이미지에서 `Traffic Light` 1개, `Speedlimit` 1개 탐지
* 일부 이미지에서는 탐지 결과 없음(no detections)

---

## 10. 결과 시각화 (여러 이미지 그리드로 보기)

다음 코드는 `/content/runs/detect/predict` 폴더의 결과 이미지를 그리드 형태로 출력합니다.

```python
import matplotlib.pyplot as plt
import cv2
import math
import os

img_dir = "/content/runs/detect/predict"
extensions = (".jpg", ".jpeg", ".png", ".bmp")

files = [f for f in os.listdir(img_dir) if f.lower().endswith(extensions)]
files = sorted(files)

cols = 4
rows = math.ceil(len(files) / cols)

plt.figure(figsize=(cols*2.2, rows*2.2))

for i, f in enumerate(files):
    path = os.path.join(img_dir, f)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(rows, cols, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f, fontsize=8)

plt.tight_layout()
plt.show()
```

---

## 11. 자주 발생할 수 있는 이슈

### 11.1 convert2Yolo requirements 설치 오류

* `convert2Yolo/requirements.txt`에는 `matplotlib==2.2.2` 등 오래된 버전이 포함되어 있습니다.
* 최신 Colab 환경에서는 해당 버전 설치 시 `metadata-generation-failed` 등의 에러가 발생할 수 있습니다.
* 이 경우:

  * 이미 Colab 기본 환경에 최신 `matplotlib`가 설치되어 있으므로
  * `pip install -r requirements.txt` 단계에서 발생하는 오류는 무시하고
  * `example.py`만 실행해도 변환이 정상적으로 이루어집니다.

---

## 12. 정리

이 프로젝트에서는 다음 과정을 통해 도로 표지판 탐지 모델을 구성하였습니다.

1. Kaggle Road Sign Detection 데이터셋 다운로드
2. VOC 형식 어노테이션을 convert2Yolo로 YOLO 포맷으로 변환
3. YOLOv8 학습용 디렉토리 구조 구성 (`images` / `labels` / train/val/test)
4. `custom_voc.yaml` 설정 파일 작성
5. `yolov8s.pt`를 기반으로 커스텀 도로 표지판 검출 모델 학습
6. 학습된 `best.pt`로 검증 및 테스트 이미지 추론
7. 결과 이미지를 그리드 형태로 시각화

위 과정을 바탕으로, 다른 객체 검출 데이터셋에도 비슷한 방식으로 **YOLOv8 커스텀 학습 파이프라인**을 쉽게 적용할 수 있습니다.

<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/f0e751a3-cc02-4c2f-aa54-9a8432040354" />
<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/f3cbc90e-826b-48e5-8e7f-0bc8b69579e6" />
<img width="2250" height="1500" alt="BoxR_curve (1)" src="https://github.com/user-attachments/assets/3177357d-105e-4765-9108-432986bfb65d" />
<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/ca131e8b-e5cf-402d-8448-b0cea1688c72" />
<img width="3000" height="2250" alt="confusion_matrix_normalized (1)" src="https://github.com/user-attachments/assets/9b9a763d-2994-4ed3-9271-5e3b3979988e" />
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/41c8b9fc-dfc6-48fb-bcd8-16eec6341fb5" />
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/be160440-d444-4f10-82aa-f3b23b0e3e9f" />

<img src="https://github.com/user-attachments/assets/5df2bd1a-0467-4923-bfd1-50c2783ef847"/>


