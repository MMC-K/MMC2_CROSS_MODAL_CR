# 학습 전후 모델 평가

같은 평가 250건·실제 이미지·무작위화된 멘션 목록을 사용해 기반 모델과 학습된 LoRA 모델을 비교한다. 이 비교는 이미지 제공 조건의 학습 전후 차이를 측정한다.

```bash
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/evaluate.sh base
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/evaluate.sh lora
python scripts/compare.py --run-dir outputs/run1
```

`base`는 Hugging Face 기반 모델이고 `lora`는 RUN_DIR의 best_adapter를 로드한다. 예측은 base_test.jsonl / lora_test.jsonl, 요약은 RESULTS.md / RESULTS.csv / RESULTS.json에 저장한다.

## 평가 지표

| 지표 | 의미 | 전체 집계 |
| --- | --- | --- |
| Exact-cluster F1 | 멘션 집합이 완전히 같은 군집의 일치 | 대화별 F1 평균 |
| B³-F1 | 각 멘션의 예측·정답 군집 겹침 | 대화별 F1 평균 |
| LEA-F1 | entity 크기로 가중한 내부 연결 복원 | 대화별 F1 평균 |
| Cross-modal link-F1 | Text–Image 멘션 쌍의 연결 복원 | 전체 쌍의 micro-F1 |

JSON 파싱 실패는 빈 예측으로 채점한다. 예측 ID를 정답에 맞춰 고치는 후처리는 적용하지 않는다. Cross-modal link-F1은 유형을 알 수 있는 Text–Image 쌍만 세므로, 잘못된 ID가 포함된 샘플 수도 따로 보고한다. B³에서는 한쪽에 빠진 멘션을 그쪽에서 단독 군집으로 간주한다.

Exact-cluster F1은 분류 정확도와 다르다. 한 군집에서 멘션 하나를 빠뜨리면 해당 군집의 완전 일치는 인정되지 않지만 B³·LEA·link-F1에는 부분적으로 복원한 연결이 반영된다. 실제 계산 예시는 성능 분석서 5.2절을 참고한다.

## 저장된 예측의 CPU 재채점

```bash
python scripts/score.py reference_results/base_test.jsonl
python scripts/score.py reference_results/lora_test.jsonl
```

학습 가중치 없이도 공개된 예측 250건과 원래 성능표가 같은지 확인할 수 있다. 직접 학습한 실행 결과는 `reference_results/`가 아니라 `outputs/`에 저장한다.
