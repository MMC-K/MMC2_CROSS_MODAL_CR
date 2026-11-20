"""Portable image-provided coreference: train, evaluate, or infer from HF weights."""
import argparse,contextlib,gc,hashlib,json,math,os,time
from pathlib import Path
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,get_cosine_schedule_with_warmup
from peft import LoraConfig,PeftModel,get_peft_model
from .common import (ROOT,MODEL_ID,REVISION,MIN_PIXELS,MAX_PIXELS,Examples,Collate,
                     optimize_patch_projection,save_json,seed_all,cuda_batch,suffix_loss)
from .metrics import parse,metrics
from .data_utils import get_coreference_prompt

def read_rows(path):
    path=Path(path)
    if path.suffix=='.json':
        data=json.loads(path.read_text());data=data if isinstance(data,list) else [data]
    else:data=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    assert data,f'Empty input: {path}'
    for i,row in enumerate(data):
        row.setdefault('sample_id',f'input_{i:05d}');row.setdefault('task','coreference');row.setdefault('modality','multimodal')
        row.setdefault('dialogue_id',row['sample_id'])
        if 'prompt' not in row:row['prompt']=get_coreference_prompt(row['history'],row['mention_pool'])
        assert row.get('image_paths'),f"Actual image_paths required: {row['sample_id']}"
        assert row['prompt'].count('[IMAGE]')==len(row['image_paths']),row['sample_id']
        known=[m['id'] for m in row['mention_pool']];assert len(known)==len(set(known))
    assert len({r['sample_id'] for r in data})==len(data),'Duplicate sample IDs'
    return data

def loader(items,processor,batch_size,training=False,sampler=None,workers=4):
    return DataLoader(Examples(items,processor,training),batch_size=batch_size,sampler=sampler,
                      num_workers=workers,pin_memory=True,persistent_workers=workers>0,
                      collate_fn=Collate(processor.tokenizer.pad_token_id))

def barrier(world):
    if world>1:dist.barrier()

def fingerprint(args,items):
    # Record model/adapter identity and input rows; scripts/check_data.py checks image bytes.
    adapter=None
    if args.adapter:
        p=Path(args.adapter)/'adapter_model.safetensors'
        adapter=dict(path=str(Path(args.adapter).resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return dict(model=args.model,revision=args.revision,adapter=adapter,max_new_tokens=args.max_new_tokens,
                data_sha256=hashlib.sha256(json.dumps(items,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
