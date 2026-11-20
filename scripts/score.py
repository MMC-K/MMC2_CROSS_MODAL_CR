#!/usr/bin/env python3
"""CPU-only scoring of previously saved predictions."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from kommerce_coref.metrics import metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    rows=[json.loads(s) for s in args.predictions.read_text().splitlines() if s.strip()]
    assert rows and all('gold' in r for r in rows),'Predictions must contain gold clusters for scoring'
    result=metrics(rows);text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text)

if __name__=='__main__':main()
