"""Same exact-cluster, B³, LEA and cross-modal-link metrics as the source experiment."""
import json

def _cluster_match(a, b):
    return set(a) == set(b)

def exact_cluster_f1(pred, gold):
    vp = [c for c in pred if isinstance(c, list)]
    vg = [c for c in gold if isinstance(c, list)]
    if not vp and not vg:
        return 1.0
    if not vp or not vg:
        return 0.0
    matched, tp = set(), 0
    for pc in vp:
        for i, gc in enumerate(vg):
            if i in matched:
                continue
            if _cluster_match(pc, gc):
                tp += 1
                matched.add(i)
                break
    p = tp / len(vp)
    r = tp / len(vg)
    return 2 * p * r / (p + r) if p + r else 0.0

def _universe(pred, gold):
    u = set()
    for c in pred + gold:
        if isinstance(c, list):
            u |= set(c)
    return u

def b3_f1(pred, gold):
    uni = _universe(pred, gold)
    if not uni:
        return 1.0  # both empty
    def mp(clusters):
        d = {}
        for c in clusters:
            if isinstance(c, list):
                for m in c:
                    d[m] = set(c)
        return d
    ps, gs = mp(pred), mp(gold)
    P = R = 0.0
    for m in uni:
        pc = ps.get(m, {m})
        gc = gs.get(m, {m})
        inter = len(pc & gc)
        P += inter / len(pc)
        R += inter / len(gc)
    n = len(uni)
    P /= n
    R /= n
    return 2 * P * R / (P + R) if P + R else 0.0

def _lea_links(n):
    # LEA (Moosavi & Strube, 2016): #coref links in a cluster; singleton -> 1
    return 1 if n == 1 else n * (n - 1) / 2.0

def _lea_recall(key, resp):
    num = den = 0.0
    for k in key:
        k = set(k)
        if not k:
            continue
        imp = len(k)
        klinks = _lea_links(len(k))
        common = 0.0
        for r in resp:
            r = set(r)
            inter = k & r
            if len(inter) > 1:
                common += _lea_links(len(inter))
            elif len(inter) == 1 and len(k) == 1 and len(r) == 1:
                common += 1.0            # correctly-resolved singleton
        num += imp * (common / klinks)
        den += imp
    return num / den if den else 0.0

def lea_f1(pred, gold):
    vp = [c for c in pred if isinstance(c, list)]
    vg = [c for c in gold if isinstance(c, list)]
    if not vp and not vg:
        return 1.0
    if not vp or not vg:
        return 0.0
    R = _lea_recall(vg, vp)
    P = _lea_recall(vp, vg)
    return 2 * P * R / (P + R) if P + R else 0.0

def parse(response):
    try:
        start,end=response.find('{'),response.rfind('}')
        assert start>=0 and end>start
        obj=json.loads(response[start:end+1]); clusters=obj['clusters']
        valid=isinstance(clusters,list) and all(isinstance(c,list) and all(isinstance(m,str) for m in c) for c in clusters)
        return (clusters,True) if valid else ([],False)
    except (ValueError,TypeError,KeyError,AssertionError):return [],False

def cross_links(clusters,types):
    return {(t,i) for c in clusters for t in c for i in c if types.get(t)=='Text' and types.get(i)=='Image'}

def metrics(predictions):
    n=len(predictions);assert n
    exact=[exact_cluster_f1(x['prediction'],x['gold']) for x in predictions]
    primary=sum(exact)/n
    assert abs(primary-sum(exact)/n)<1e-12
    tp=npred=ngold=0
    for item in predictions:
        p=cross_links(item['prediction'],item['mention_types']);g=cross_links(item['gold'],item['mention_types'])
        tp+=len(p&g);npred+=len(p);ngold+=len(g)
    p=tp/npred if npred else 0;r=tp/ngold if ngold else 0
    return dict(n=n,exact_cluster_f1=primary,b3_f1=sum(b3_f1(x['prediction'],x['gold']) for x in predictions)/n,
                lea_f1=sum(lea_f1(x['prediction'],x['gold']) for x in predictions)/n,
                cross_modal_link_precision=p,cross_modal_link_recall=r,cross_modal_link_f1=2*p*r/(p+r) if p+r else 0,
                valid_json=sum(x['valid_json'] for x in predictions),empty_predictions=sum(not x['prediction'] for x in predictions),
                gold_empty=sum(not x['gold'] for x in predictions),
                unknown_mention_ids=sum(any(m not in x['mention_types'] for c in x['prediction'] for m in c) for x in predictions),
                token_limit_hits=sum(x['output_tokens']>=512 for x in predictions))
