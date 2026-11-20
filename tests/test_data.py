import copy,json,unittest
from pathlib import Path
from kommerce_coref.data_utils import transform,clusters_equal
ROOT=Path(__file__).resolve().parents[1]
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
class DataTests(unittest.TestCase):
    def test_dialogue_splits_are_disjoint(self):
        ids=[]
        for split,n in [('train',2500),('dev',500),('test',250)]:
            r=rows(ROOT/'data'/f'{split}.jsonl');s={x['dialogue_id'] for x in r}
            self.assertEqual(len(r),n);self.assertEqual(len(s),n);ids.append(s)
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    def test_permutation_does_not_depend_on_gold(self):
        row=rows(ROOT/'data/unshuffled/test.jsonl')[0];other=copy.deepcopy(row)
        other['gold']=[[row['mention_pool'][0]['id'],row['mention_pool'][1]['id']]]
        a,ma=transform(row);b,mb=transform(other)
        self.assertEqual(ma,mb);self.assertEqual(a['mention_pool'],b['mention_pool']);self.assertEqual(a['prompt'],b['prompt'])
    def test_inverse_mapping_preserves_reference_groups(self):
        row=rows(ROOT/'data/unshuffled/test.jsonl')[0];changed,m=transform(row)
        inverse={v:k for k,v in m['old_to_new'].items()}
        self.assertTrue(clusters_equal([[inverse[x] for x in c] for c in changed['gold']],row['gold']))
        self.assertEqual(changed['image_paths'],row['image_paths'])
    def test_inference_example_has_no_gold_and_has_real_images(self):
        row=rows(ROOT/'examples/inference_sample.jsonl')[0]
        self.assertNotIn('gold',row)
        self.assertEqual(row['prompt'].count('[IMAGE]'),len(row['image_paths']))
        for name in row['image_paths']:self.assertTrue((ROOT/name).is_file())
if __name__=='__main__':unittest.main()
