# 실행 환경과 설치

기준 실험은 Linux, Python 3.12, NVIDIA A100 80GB GPU 2대에서 수행하였다. 기반 모델은 Hugging Face에서 내려받으므로 최초 다운로드에는 인터넷이 필요하다. Python 환경·모델·학습 산출물을 위한 여유 공간은 약 30GB를 권장한다.

```bash
git clone https://github.com/MMC-K/MMC2_CROSS_MODAL_CR.git
cd MMC2_CROSS_MODAL_CR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/check_data.py
python scripts/download_model.py
```

CUDA 12.8 wheel과 호환되는 NVIDIA driver가 필요하다. 다른 CUDA 설치는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)를 따른다. 재현 기준의 나머지 패키지는 requirements.txt에 고정하였다.

기본 GPU는 물리 1·2번이다. `GPUS=1`처럼 한 개 GPU로도 실행할 수 있다. 다른 번호의 서버에서는 `GPUS` 값을 바꾼다. Hugging Face 캐시 위치는 `HF_HOME`으로 지정할 수 있고, 가중치는 Git 저장소에 저장하지 않는다.

## CPU에서 확인 가능한 항목

학습·추론에는 CUDA GPU가 필요하다. 데이터 확인, 무작위화 재생성, 저장된 예측 채점은 CPU에서 수행할 수 있다.

```bash
python -m pip install Pillow==12.1.1
python scripts/check_data.py
python scripts/prepare_data.py
python -m unittest discover -s tests -v
python scripts/score.py reference_results/lora_test.jsonl
```
