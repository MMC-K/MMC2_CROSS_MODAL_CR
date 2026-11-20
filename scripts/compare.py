#!/usr/bin/env python3
"""Compare matching image-provided base-model and LoRA predictions."""
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from kommerce_coref.metrics import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/run1');args=p.parse_args()
    result={};items={}
    for name in ['base_test','lora_test']:
        items[name]=[json.loads(s) for s in (args.run_dir/(name+'.jsonl')).read_text().splitlines()]
        result[name]=metrics(items[name])
    assert [(r['sample_id'],r['gold']) for r in items['base_test']]==[(r['sample_id'],r['gold']) for r in items['lora_test']]
    lines=['# 이미지 제공 조건: 학습 전후 평가','', '| 모델 | Exact-cluster F1 | B³-F1 | LEA-F1 | Cross-modal link-F1 | JSON 유효 |','|---|---:|---:|---:|---:|---:|']
    for name,label in [('base_test','학습 전'),('lora_test','LoRA 학습 후')]:
        r=result[name];lines.append(f"| {label} | {r['exact_cluster_f1']:.4f} | {r['b3_f1']:.4f} | {r['lea_f1']:.4f} | {r['cross_modal_link_f1']:.4f} | {r['valid_json']}/{r['n']} |")
    text='\n'.join(lines)+'\n';(args.run_dir/'RESULTS.md').write_text(text)
    (args.run_dir/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    keys=['exact_cluster_f1','b3_f1','lea_f1','cross_modal_link_f1','valid_json','unknown_mention_ids']
    with (args.run_dir/'RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['model']+keys);w.writeheader()
        for name,r in result.items():w.writerow(dict(model=name,**{k:r[k] for k in keys}))
    print(text)

if __name__=='__main__':main()
