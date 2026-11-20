# 2B 모델의 LoRA 학습

기반 모델의 언어 모델 attention과 MLP에 LoRA를 추가한다. 기반 가중치·vision encoder·visual merger는 고정하며 LoRA 파라미터 17,432,576개를 학습한다. 입력 prompt는 loss에서 제외하고 정답 cluster JSON과 EOS token에 대해서만 cross entropy를 계산한다.

```bash
python scripts/check_data.py
python scripts/download_model.py
GPUS=1,2 RUN_DIR=outputs/run1 bash scripts/train.sh
```

검증 exact-cluster F1이 가장 높은 adapter를 `outputs/run1/best_adapter/`에 저장한다. 동률이면 앞 epoch를 유지한다. Test는 checkpoint 선택에 사용하지 않는다.

| 항목 | 기준 설정 |
| --- | --- |
| 모델 / revision | Qwen/Qwen3-VL-2B-Instruct / 89644892e4d85e24eaac8bacfd4f463576704203 |
| 학습 / 검증 | 2,500 / 500건 |
| 문맥 | 마지막 16개 turn |
| LoRA | r=16, alpha=32, dropout=0.05 |
| 학습 모듈 | language-model attention q/k/v/o, MLP gate/up/down |
| GPU | A100 80GB 2대, 물리 GPU 1·2 |
| Epoch / seed | 3 / 42; GPU별 dropout seed 42·43 |
| Batch | GPU별 16, 전체 32; 마지막 잔여 batch 4 |
| Optimizer | AdamW, lr=1e-4, weight decay=0.01 |
| Schedule | cosine, warmup 3%, gradient clipping 1 |
| 정밀도 | BF16 autocast, FP32 LoRA, gradient checkpointing |
| 이미지 | 이미지당 64~256 visual tokens |
| 출력 | greedy, 최대 512 new tokens |

## 메모리와 소규모 점검

GPU별 micro batch를 줄이면 gradient accumulation으로 전체 batch 32를 유지한다. GPU 1개 또는 2개를 지원한다.

```bash
GPUS=1,2 MICRO_BATCH=2 EVAL_BATCH=2 RUN_DIR=outputs/small_batch bash scripts/train.sh
GPUS=1 MICRO_BATCH=2 EVAL_BATCH=2 RUN_DIR=outputs/single_gpu bash scripts/train.sh
```

전체 학습 전 실행 흐름을 확인하는 예시이다. 이 점수는 동작 점검용이며 기준 성능표와 비교하지 않는다.

```bash
GPUS=1,2 MICRO_BATCH=2 EVAL_BATCH=2 RUN_DIR=outputs/smoke   bash scripts/train.sh --limit-train 8 --limit-dev 4 --epochs 1 --warmup-ratio 0 --workers 0
GPUS=1,2 EVAL_BATCH=2 RUN_DIR=outputs/smoke   bash scripts/evaluate.sh lora --limit-input 4 --workers 0
```

기존 학습 폴더를 덮어쓰지 않으므로 새 실행에는 새 RUN_DIR을 사용한다. 이 배포판은 학습 도중의 자동 재개 기능을 제공하지 않는다. 모델 다운로드 후 캐시만 사용하려면 실행 명령 뒤에 `--local-files-only`를 추가한다.
