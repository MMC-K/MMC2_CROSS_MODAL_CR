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
