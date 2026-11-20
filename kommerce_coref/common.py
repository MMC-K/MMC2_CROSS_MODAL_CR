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

def optimize_patch_projection(model):
    # Each Conv3d input is exactly one non-overlapping (2,16,16) patch.
    # Flattening its kernel gives the same linear map without slow cuDNN
    # Conv3d algorithm initialization on this A100/CUDA installation.
    patch = model.model.visual.patch_embed
    def forward(self, hidden_states):
        weight = self.proj.weight
        flat = hidden_states.reshape(-1, weight[0].numel()).to(weight.dtype)
        return torch.nn.functional.linear(flat, weight.flatten(1), self.proj.bias)
    patch.forward = types.MethodType(forward, patch)

def chat_prefix(prompt):
    body = prompt.replace('[IMAGE]', '<|vision_start|><|image_pad|><|vision_end|>')
    return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + body + '<|im_end|>\n<|im_start|>assistant\n'

class Examples(Dataset):
    def __init__(self, rows, processor, training=False):
        self.rows, self.processor, self.training = rows, processor, training
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        images = []
        for filename in row['image_paths']:
            path = Path(filename)
            if not path.is_absolute(): path = ROOT / path
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        assert row['prompt'].count('[IMAGE]') == len(images), row['sample_id']
        kw = dict(text=chat_prefix(row['prompt']), return_tensors='pt')
        if images: kw['images'] = images
        enc = dict(self.processor(**kw))
        ids = enc['input_ids'][0]
        enc.pop('attention_mask', None)
        if self.training:
            answer = json.dumps({'clusters': row['gold']}, ensure_ascii=False) + '<|im_end|>'
            target = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            enc['labels'] = torch.tensor([-100] * len(ids) + target)
            ids = torch.cat([ids, torch.tensor(target)])
            enc['answer_length'] = len(target)
        enc['input_ids'] = ids
        assert len(ids)<=8192, row['sample_id']
        if self.training: assert enc['answer_length']<512, row['sample_id']
        enc['row_index'] = i
        return enc

class Collate:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, examples):
        length = max(len(x['input_ids']) for x in examples)
        ids = torch.full((len(examples), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100) if 'labels' in examples[0] else None
        for i, x in enumerate(examples):
            n = len(x['input_ids']); ids[i, -n:] = x['input_ids']; mask[i, -n:] = 1
            if labels is not None: labels[i, -n:] = x['labels']
        batch = dict(input_ids=ids, attention_mask=mask)
        for key in ['pixel_values', 'image_grid_thw']:
            values = [x[key] for x in examples if key in x]
            if values: batch[key] = torch.cat(values)
        if labels is not None:
            batch['labels'] = labels
            batch['answer_length'] = max(x['answer_length'] for x in examples)
        batch['row_indices'] = [x['row_index'] for x in examples]
        return batch

def cuda_batch(batch):
    return {k: v.to('cuda', non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

@torch.autocast('cuda', dtype=torch.bfloat16)
def suffix_loss(model, batch):
    labels = batch.pop('labels'); n = batch.pop('answer_length')
    output = model(**batch, use_cache=False, logits_to_keep=n+1)
    return torch.nn.functional.cross_entropy(output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]), labels[:, -n:].reshape(-1))
