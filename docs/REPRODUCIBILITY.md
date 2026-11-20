# 재현 절차와 검증 범위

이 저장소는 원래 실험에서 실행한 2B LoRA 학습·평가 코드를 독립 실행 형태로 제공한다. 기반 모델 revision, 데이터·이미지 해시, 대화 분할, 멘션 변환 기록과 실제 예측을 함께 보존한다.

## 전체 실행

README의 설치 후 다음 명령으로 데이터 확인부터 학습 전후 비교까지 수행한다.

```bash
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/run_all.sh
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/infer.sh
```

## GPU 없이 기준 결과 점검

데이터·멘션 변환·평가 지표 검사는 `bash scripts/check_all.sh`로 한 번에 실행할 수 있다. 개별 명령은 다음과 같다.

```bash
python scripts/check_data.py
python scripts/prepare_data.py
python -m unittest discover -s tests -v
python scripts/score.py reference_results/base_test.jsonl
python scripts/score.py reference_results/lora_test.jsonl
```

`DATA_SHA256.json`은 학습·검증·평가 JSONL과 필요한 이미지의 해시이다. `ARTIFACT_SHA256.json`은 공개 저장소의 파일 해시 목록이며 자기 자신은 제외한다. 분석서 PDF·Word는 이 프로젝트의 최종본과 같은 파일이다.

## 검증 기록의 구분

`reference_results/PACKAGE_VALIDATION.json`은 2026-11-13 공유 패키지의 검증 기록이다. 이번 GitHub 공개본의 CPU 검사·소규모 GPU 학습·평가·추론 확인은 `docs/RELEASE_VALIDATION.json`에 별도로 기록한다. 소규모 실행은 학습 흐름을 확인하며 3-epoch 전체 학습을 다시 수행한 결과로 해석하지 않는다.

원래 학습 설정은 `provenance/config.json`, 최종 선택은 `reference_results/selection.json`, epoch별 검증 점수와 237 optimizer update는 `reference_results/`에서 확인한다. 학습 seed, 입력 표현 및 이미지 제공 조건을 유지해 비교한다.
