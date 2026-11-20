# 이미지 제공 조건의 학습 전후 성능

Qwen3-VL-2B-Instruct, 학습 2,500 / 검증 500 / 평가 250개 대화, k=16, 3 epoch, seed 42.
후보 ID와 표시 순서는 정답 cluster와 독립적으로 무작위화했다. 학습·검증·평가에 모두 실제 이미지를 제공한다.

| 모델 | Exact-cluster F1 | B³-F1 | LEA-F1 | Cross-modal link-F1 |
|---|---:|---:|---:|---:|
| 학습 전 | 0.0000 | 0.3850 | 0.0594 | 0.2324 |
| LoRA 학습 후 | 0.9187 | 0.9801 | 0.9689 | 0.9838 |

검증 점수로 선택한 epoch 3의 adapter를 평가했다. 평가 대화·이미지·prompt template·greedy decoding 설정은 학습 전후 동일하다.
주어진 텍스트 및 이미지 영역 mention 후보를 cluster로 묶는 태스크이며 mention 검출은 포함하지 않는다.
학습 전: JSON 유효 221/250, 출력 상한 도달 29건. 학습 후: JSON 유효 250/250, 잘못된 mention ID 0건.
