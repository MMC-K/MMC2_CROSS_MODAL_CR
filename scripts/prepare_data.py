#!/usr/bin/env python3
"""Reconstruct the bundled randomized data from the included unshuffled splits."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from kommerce_coref.data_utils import transform,audit

def main():
    manifest=json.loads((ROOT/'DATA_SHA256.json').read_text());result={}
    for split in ['train','dev','test']:
        original=[json.loads(s) for s in (ROOT/'data/unshuffled'/f'{split}.jsonl').read_text().splitlines()]
        derived=[transform(row)[0] for row in original]
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in derived)
        name=f'data/{split}.jsonl'
        assert hashlib.sha256(text.encode()).hexdigest()==manifest[name],name
        (ROOT/name).write_text(text)
        result[split]=audit(derived)
    print(json.dumps(dict(status='reproduced',randomization=result),indent=2))

if __name__=='__main__':main()
