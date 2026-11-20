import unittest
from kommerce_coref.metrics import exact_cluster_f1,b3_f1,lea_f1,parse,cross_links,metrics

class MetricTests(unittest.TestCase):
    gold=[['M3','M8','M13'],['M4','M5']]
    pred=[['M3','M8'],['M4','M5']]
    types={'M3':'Image','M8':'Text','M13':'Text','M4':'Text','M5':'Image'}
    def test_cluster_order_does_not_change_match(self):
        self.assertEqual(exact_cluster_f1([['M5','M4'],['M13','M3','M8']],self.gold),1)
    def test_duplicate_prediction_is_not_an_extra_true_positive(self):
        self.assertAlmostEqual(exact_cluster_f1([['a','b'],['a','b']],[['a','b']]),2/3)
    def test_one_missing_mention_changes_complete_cluster_score(self):
        self.assertEqual(exact_cluster_f1(self.pred,self.gold),.5)
        self.assertAlmostEqual(b3_f1(self.pred,self.gold),11/13)
        self.assertAlmostEqual(lea_f1(self.pred,self.gold),.75)
    def test_parser_accepts_wrapped_valid_cluster_json(self):
        self.assertEqual(parse('답: {"clusters": [["a", "b"]]}'),([['a','b']],True))
    def test_parser_rejects_truncation_and_wrong_id_types(self):
        for text in ['{"clusters": [','{"clusters": [[1, 2]]}','{"other": []}']:
            self.assertEqual(parse(text),([],False))
    def test_cross_links_use_only_text_image_pairs(self):
        self.assertEqual(cross_links(self.gold,self.types),{('M8','M3'),('M13','M3'),('M4','M5')})
    def row(self,gold,pred,types=None):
        return dict(gold=gold,prediction=pred,mention_types=types or self.types,valid_json=True,output_tokens=30)
    def test_link_score_is_corpus_micro_not_mean_sample_f1(self):
        rows=[self.row(self.gold,self.pred),self.row([['M4','M5']],[])]
        self.assertAlmostEqual(metrics(rows)['cross_modal_link_f1'],2/3)
    def test_unknown_ids_are_counted_separately(self):
        r=metrics([self.row(self.gold,[['M8','M3','unknown']])])
        self.assertEqual(r['unknown_mention_ids'],1)
    def test_empty_prediction_scores_zero_exact_and_links(self):
        r=metrics([self.row(self.gold,[])])
        self.assertEqual(r['exact_cluster_f1'],0)
        self.assertEqual(r['cross_modal_link_f1'],0)

if __name__=='__main__':unittest.main()
