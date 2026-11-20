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
