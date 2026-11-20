# 데이터 포맷

KoMMERCE에서 추출한 학습 2,500건·검증 500건·평가 250건을 포함한다. 대화 ID는 세 분할 사이에 겹치지 않는다. 대화당 마지막 16개 발화의 window 하나를 사용하며 모든 샘플에 실제 이미지와 Text–Image 정답 군집이 있다. 필요한 고유 이미지 3,980장은 `data/images/`에 포함한다.

## JSONL 한 줄의 구성

| 필드 | 의미 | 모델 입력 여부 |
| --- | --- | --- |
| sample_id | 평가 샘플의 고유 식별자 | 관리용 |
| dialogue_id | 원본 대화 ID | 관리용 |
| prompt | 과제 지시·16개 발화·멘션 목록·응답 형식 | 입력 |
| image_paths | prompt의 [IMAGE] 순서와 같은 로컬 이미지 경로 | 입력 |
| image_urls | 원본 이미지 출처 URL | 출처 정보 |
| mention_pool | id, turn_idx, type, content로 구성한 멘션 목록 | prompt에 반영 |
| gold | 정답 멘션 군집 목록 | 학습 정답 / 평가 채점용 |

텍스트 멘션의 content는 실제 문자열이고 이미지 멘션은 `Bbox: [x, y, w, h]` 형식의 정규화된 영역 좌표이다. 이미지는 영역을 잘라낸 사진이 아닌 전체 이미지를 제공한다. `image_paths`는 저장소 루트 기준 상대 경로이다. 외부 이미지 서버에 다시 접속할 필요가 없다.

모델은 입력의 M… ID를 사용하여 `{"clusters": [["M4", "M5"], ...]}` 형식으로 출력한다. 원본 데이터의 entity 묶음은 gold로만 변환하며 입력에 넣지 않는다. 원본 JSON과 모델용 JSONL의 대응은 성능 분석서 부록 B에 있다.

## ID와 표시 순서의 무작위화

원본 entity 의미를 유지하면서 멘션 ID 재배정과 목록 표시 순서를 별도의 난수열로 무작위화하였다. 난수열은 seed 42·대화 ID·변환 종류에서 결정하며 정답 군집은 난수 생성에 사용하지 않는다. `data/unshuffled/`와 `provenance/id_maps.json`은 같은 샘플의 변환 전 표현과 대응표이다.

```bash
python scripts/prepare_data.py
python scripts/check_data.py
```

첫 명령은 포함된 변환 전 데이터에서 JSONL을 재생성하고 기존 DATA_SHA256.json과 같은지 검증한다. 두 번째 명령은 대화 분리, 실제 이미지의 디코딩과 파일 해시를 검사한다.

## 데이터 출처

데이터셋: [KoMMERCE](https://github.com/MMC-K/multimodal_fashion_dialog_dataset). 이미지는 대화에서 사용한 K-Fashion 이미지이며 원본 URL을 각 샘플에 보존하였다. 전체 15,293개 코퍼스 중 이번 실험의 3,250개 window를 포함한다. 데이터·이미지의 이용 조건은 해당 원출처의 안내를 따른다.
