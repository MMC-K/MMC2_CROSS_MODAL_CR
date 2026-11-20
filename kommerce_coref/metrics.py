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
