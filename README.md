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

# 모델 학습

```bash
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/train.sh
```

이 명령은 LoRA 학습과 epoch별 검증을 수행한다. 검증 exact-cluster F1이 가장 높은 adapter는 `outputs/run1/best_adapter/`에 저장된다. 새 실행에는 새 RUN_DIR을 지정한다.

메모리를 줄이는 예시는 다음과 같다. 전체 batch 32는 gradient accumulation으로 유지한다.

```bash
GPUS=1,2 MICRO_BATCH=2 EVAL_BATCH=2 RUN_DIR=outputs/small_batch bash scripts/train.sh
```

[학습 안내](docs/TRAINING.md)에 전체 설정, GPU 1개 실행, 소규모 학습 점검 명령을 정리했다.

# 모델 평가

학습을 마친 뒤 같은 RUN_DIR로 기반 모델과 학습된 모델을 각각 평가한다.

```bash
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/evaluate.sh base
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/evaluate.sh lora
python scripts/compare.py --run-dir outputs/run1
```

예측은 `base_test.jsonl`, `lora_test.jsonl`에 저장되며 성능표는 `outputs/run1/RESULTS.md`, `RESULTS.csv`, `RESULTS.json`에 저장된다. 평가 조건과 지표는 [평가 안내](docs/EVALUATION.md)를 따른다.

기존 실험의 공개 예측은 GPU나 가중치 없이 다시 채점할 수 있다.

```bash
python scripts/score.py reference_results/base_test.jsonl
python scripts/score.py reference_results/lora_test.jsonl
```

# 모델 추론

학습한 adapter로 정답 없는 예제의 군집을 생성한다.

```bash
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/infer.sh
```

예측은 `outputs/run1/inference/predictions.jsonl`에 저장된다. 사용자 대화·이미지를 입력하는 방법은 [추론 안내](docs/INFERENCE.md)에 있다.

# 학습부터 평가까지 한 번에 실행

```bash
GPUS=1,2 RUN_DIR=outputs/run2 bash scripts/run_all.sh
```

데이터 검사 → 기반 모델 다운로드 → 학습·검증 → 학습 전후 평가 → 성능표 생성을 순서대로 수행한다.

# 실험 결과

| 모델 | Exact-cluster F1 | B³-F1 | LEA-F1 | Cross-modal link-F1 |
| --- | ---: | ---: | ---: | ---: |
| 학습 전 | 0.0000 | 0.3850 | 0.0594 | 0.2324 |
| LoRA 학습 후 | **0.9187** | **0.9801** | **0.9689** | **0.9838** |

모든 점수는 0~1 범위이다. Exact-cluster F1은 군집 전체의 완전 일치, B³·LEA는 부분적인 군집 복원, Cross-modal link-F1은 텍스트·이미지 멘션 사이 연결을 측정한다. 상세 계산과 집계 방식은 [평가 안내](docs/EVALUATION.md)에 있다.

학습 후 250건 모두 유효한 JSON을 출력했다. 학습 전 JSON 파싱 실패는 29건, 잘못된 멘션 ID가 포함된 샘플은 180건이었으며 학습 후에는 두 항목 모두 0건이었다. 학습 전 exact-cluster F1이 0인 것은 부분적인 연결도 전혀 맞히지 못했다는 뜻은 아니다.

[전체 결과와 해석](docs/RESULTS.md), [모델 구성](docs/MODEL_CARD.md), `reference_results/`의 실제 예측·검증 지표를 함께 제공한다. 수치는 한 seed의 기준 실험 결과이며 실행 환경에 따라 재학습 결과가 달라질 수 있다.

# 성능 분석서

이 프로젝트에서 작성한 조병길 연구원(KETI)의 기술문서를 함께 제공한다. 전체 대화·모델 입력과 출력·데이터 구조·지표 계산·성공 및 오류 사례를 포함한다.

- [성능 분석서 PDF](docs/reports/coreference_performance_analysis_v1.pdf)
- [성능 분석서 Word](docs/reports/coreference_performance_analysis_v1.docx)
- [분석서와 저장소 실행 방법의 연결](docs/REPORT.md)
