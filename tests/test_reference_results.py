import json,unittest
from pathlib import Path
from kommerce_coref.metrics import metrics
ROOT=Path(__file__).resolve().parents[1]
class ReferenceTests(unittest.TestCase):
    def test_published_predictions_reproduce_all_reported_metrics(self):
        expected=json.loads((ROOT/'reference_results/IMAGE_ON_RESULTS.json').read_text())['metrics']
        for name,key in [('base_test','zero_shot'),('lora_test','lora')]:
            rows=[json.loads(s) for s in (ROOT/'reference_results'/f'{name}.jsonl').read_text().splitlines()]
            self.assertEqual(metrics(rows),expected[key])
    def test_base_and_lora_use_identical_samples_and_gold(self):
        values=[]
        for name in ['base_test','lora_test']:
            rows=[json.loads(s) for s in (ROOT/'reference_results'/f'{name}.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),250);values.append([(r['sample_id'],r['gold']) for r in rows])
        self.assertEqual(*values)
if __name__=='__main__':unittest.main()
