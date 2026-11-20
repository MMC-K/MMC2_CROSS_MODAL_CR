"""Model/data helpers adapted from the verified local LoRA implementation."""
import json,random,types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[1]
MODEL_ID='Qwen/Qwen3-VL-2B-Instruct'
REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
MIN_PIXELS=64*32*32
MAX_PIXELS=256*32*32

def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
