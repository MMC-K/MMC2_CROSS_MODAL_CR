#!/usr/bin/env python3
"""Download the pinned HF foundation model into the external Hugging Face cache."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
# Do not import CUDA/model modules just to download files.
MODEL='Qwen/Qwen3-VL-2B-Instruct'
REVISION='89644892e4d85e24eaac8bacfd4f463576704203'

def main():
    from huggingface_hub import snapshot_download
    p=argparse.ArgumentParser();p.add_argument('--cache-dir');p.add_argument('--local-files-only',action='store_true');args=p.parse_args()
    path=snapshot_download(MODEL,revision=REVISION,cache_dir=args.cache_dir,local_files_only=args.local_files_only,token=False,
        allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.jinja','LICENSE*'])
    print(path)

if __name__=='__main__':main()
