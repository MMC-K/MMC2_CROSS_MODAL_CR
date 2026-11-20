# 대화 내 크로스 모달 상호 참조 해결 모델 V.1 (SW)

Qwen3-VL-2B-Instruct 기반 크로스 모달 상호 참조 해결 모델의 학습·평가·추론 코드입니다.

# 문제 설명

발화와 이미지가 포함된 대화에서, “이 상품”, “이거” 같은 텍스트 표현이 이미지 속 어느 상품을 가리키는지 연결한다. 모델은 주어진 텍스트·이미지 영역 멘션을 같은 참조 대상별 군집(cluster)으로 묶어 JSON으로 출력한다.

```text
Turn 10 (system): [검은색 브라탑 이미지: M5]
Turn 11 (system): 이 브라탑은 어떠세요?     ← M4
Turn 15 (system): [골드 브라탑 이미지: M3]
Turn 16 (system): 이 골드 색상의 브라탑은 어떠세요? ← M13
Turn 17 (user): 오, 이거 너무 예쁘다!       ← M8
```

```json
{"clusters": [["M3", "M8", "M13"], ["M4", "M5"]]}
```

텍스트 멘션과 이미지 영역 좌표는 입력으로 제공한다. [문제 정의](docs/TASK.md)와 [데이터 포맷](docs/DATA_FORMAT.md)에 과제와 입력·정답의 구성을 설명했다.

# 실험 설정

- 데이터셋: [KoMMERCE](https://github.com/MMC-K/multimodal_fashion_dialog_dataset)의 학습 **2,500건**, 검증 **500건**, 평가 **250건**.
- 실제 이미지 **3,980장**을 포함하며, 학습·검증·평가 대화 ID는 서로 겹치지 않는다.
- 모델: **Qwen/Qwen3-VL-2B-Instruct**, revision `89644892e4d85e24eaac8bacfd4f463576704203`.
- 학습 방식: 언어 모델의 LoRA(r=16, alpha=32, dropout=0.05); 기반 가중치·vision encoder·visual merger 고정.
- 환경: Python 3.12, A100 80GB 2대, 기본 물리 GPU **1·2번**.
- 설정: 마지막 **16개 turn**, 3 epochs, seed 42, 전체 batch 32, AdamW lr=1e-4, BF16.
- 비교: 실제 이미지를 제공한 동일 평가 250건에서 **학습 전후 성능**을 측정한다.

멘션 ID와 표시 순서는 정답 군집과 독립적으로 무작위화했다. 최종 adapter는 검증 exact-cluster F1로 선택하며 Test는 선택에 사용하지 않는다.

# 필요 패키지 설치

```bash
git clone https://github.com/MMC-K/MMC2_CROSS_MODAL_CR.git
cd MMC2_CROSS_MODAL_CR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CUDA 12.8 호환 NVIDIA driver가 필요하다. [설치 안내](docs/INSTALL.md)에 다른 GPU 구성과 CPU 점검 방법을 설명했다.

# 데이터 확인 및 모델 다운로드

```bash
python scripts/check_data.py
python scripts/download_model.py
```

대화·정답·이미지는 저장소에 포함한다. 기반 모델과 학습된 LoRA 가중치는 포함하지 않으며, Hugging Face에서 기반 모델을 받아 학습한다. 최초 모델 다운로드에는 인터넷이 필요하다. `HF_HOME`으로 모델 캐시 위치를 지정할 수 있다.
