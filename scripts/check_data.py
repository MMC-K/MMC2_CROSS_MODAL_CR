#!/usr/bin/env python3
"""Validate the bundled image-provided train/dev/test data and image checksums."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-checksums',action='store_true');args=p.parse_args()
    expected=dict(train=2500,dev=500,test=250);ids={};images=set();counts={}
    for split,n in expected.items():
        rows=[json.loads(s) for s in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
        assert len(rows)==n;ids[split]={r['dialogue_id'] for r in rows};assert len(ids[split])==n
        for r in rows:
            assert r['image_paths'] and r['prompt'].count('[IMAGE]')==len(r['image_paths'])
            pool={m['id']:m for m in r['mention_pool']};assert len(pool)==len(r['mention_pool'])
            assert r['gold']
            for c in r['gold']:
                assert len(c)==len(set(c)) and {pool[mid]['type'] for mid in c}=={'Text','Image'}
            for name in r['image_paths']:
                path=ROOT/name;assert path.is_file(),path;images.add(path)
        counts[split]=len(rows)
    assert not(ids['train']&ids['dev'] or ids['train']&ids['test'] or ids['dev']&ids['test'])
    for path in images:
        with Image.open(path) as im:im.verify()
    if not args.skip_checksums:
        manifest=json.loads((ROOT/'DATA_SHA256.json').read_text())
        for name,want in manifest.items():assert sha(ROOT/name)==want,name
    print(json.dumps(dict(status='ok',splits=counts,unique_images=len(images),dialogue_disjoint=True,checksums=not args.skip_checksums),ensure_ascii=False))

if __name__=='__main__':main()
