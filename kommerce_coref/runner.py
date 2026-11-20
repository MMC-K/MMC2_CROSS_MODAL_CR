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

@torch.inference_mode()
@torch.autocast('cuda',dtype=torch.bfloat16)
def evaluate(model,processor,items,out,name,rank,world,args,has_gold=True):
    model.eval();shard=items[rank::world];path=out/f'{name}.rank{rank}.jsonl';start=time.time()
    assert not path.exists(),f'Existing predictions: {path}; choose a new --output-dir/--name.'
    with path.open('w') as stream:
        for batch in loader(shard,processor,args.eval_batch,workers=args.workers):
            indices=batch.pop('row_indices');batch=cuda_batch(batch)
            generated=model.generate(**batch,do_sample=False,max_new_tokens=args.max_new_tokens,use_cache=True,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            output_ids=generated[:,batch['input_ids'].shape[1]:]
            responses=processor.batch_decode(output_ids,skip_special_tokens=True)
            for j,(i,response) in enumerate(zip(indices,responses)):
                row=shard[i];prediction,valid=parse(response)
                item={k:row[k] for k in ['sample_id','dialogue_id','task','modality']}
                if has_gold:item['gold']=row['gold']
                item.update(prediction=prediction,response=response,valid_json=valid,
                            output_tokens=sum(t!=processor.tokenizer.pad_token_id for t in output_ids[j].tolist()),
                            mention_types={m['id']:m['type'] for m in row['mention_pool']})
                stream.write(json.dumps(item,ensure_ascii=False)+'\n')
            stream.flush()
    barrier(world);result=None
    if rank==0:
        by_id={}
        for k in range(world):
            for line in (out/f'{name}.rank{k}.jsonl').read_text().splitlines():
                row=json.loads(line);assert row['sample_id'] not in by_id;by_id[row['sample_id']]=row
        assert len(by_id)==len(items)
        merged=[by_id[r['sample_id']] for r in items]
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
        if has_gold:
            result=metrics(merged);save_json(out/f'{name}.metrics.json',result)
        else:result=dict(n=len(merged),valid_json=sum(r['valid_json'] for r in merged))
        print(json.dumps(dict(event='evaluation',name=name,seconds=round(time.time()-start,1),metrics=result)),flush=True)
    if world>1:
        box=[result];dist.broadcast_object_list(box,src=0);result=box[0]
    return result

def initialize(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for training/inference. CPU data checking and scoring are separate scripts.')
    rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    assert world in [1,2],'Supported: one or two CUDA GPUs'
    torch.cuda.set_device(local);torch.set_num_threads(4)
    if world>1:
        from datetime import timedelta
        dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    seed_all(args.seed)
    # Resolve the pinned repository once; load optional processor files locally too.
    # This avoids fallback lookups on the model's moving main branch.
    if Path(args.model).is_dir():
        resolved_model=args.model
    else:
        from huggingface_hub import snapshot_download
        resolved_model=snapshot_download(args.model,revision=args.revision,local_files_only=args.local_files_only,token=False,
            allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.jinja','LICENSE*'])
    kw=dict(local_files_only=True)
    processor=AutoProcessor.from_pretrained(resolved_model,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS,**kw)
    processor.image_processor.size={'shortest_edge':MIN_PIXELS,'longest_edge':MAX_PIXELS};processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(resolved_model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',**kw).cuda()
    optimize_patch_projection(model)
    return rank,world,processor,model
