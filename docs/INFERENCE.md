# 새로운 대화의 군집 추론

학습된 adapter를 로드하고 정답이 없는 대화에서 cluster JSON을 생성한다. 예제 입력은 실제 평가 예시에서 gold를 제거한 파일이다.

```bash
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/infer.sh
```

예측은 `outputs/run1/inference/predictions.jsonl`에 저장한다. 다른 입력을 사용하는 경우:

```bash
GPUS=1 RUN_DIR=outputs/run1 bash scripts/infer.sh   --input examples/my_input.jsonl --output-dir outputs/my_inference
```

JSONL의 각 줄에는 다음 필드가 필요하다.

| 필드 | 내용 |
| --- | --- |
| sample_id | 입력 식별자 |
| prompt 또는 history | task prompt 또는 `Turn 0 (user): ...` 형식의 발화 문자열 배열 |
| image_paths | [IMAGE] 표시 순서와 같은 이미지 경로 |
| mention_pool | 각 멘션의 id, turn_idx, type(Text/Image), content |

`history`만 제공하면 같은 template으로 prompt를 생성한다. 텍스트 멘션 구간과 이미지 영역 좌표는 입력으로 제공해야 한다. 정답 gold는 추론에 필요하지 않다.

이미지 경로는 저장소 루트 기준 상대 경로 또는 절대 경로를 사용할 수 있다. prompt의 `[IMAGE]` 개수와 image_paths 길이가 같아야 한다. 저장된 `prediction`은 `[["M4", "M5"], ...]` 구조이며 response에는 모델이 생성한 원문 JSON이 들어 있다.
