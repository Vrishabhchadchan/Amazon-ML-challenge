"""Macro F0.5 exactly as the challenge defines it (per S1 entity, then mean)."""
import numpy as np


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map):
    """Both maps: s1_id -> set of ids. Scored over the keys of true_map."""
    return float(np.mean([f05(pred_map.get(k, set()), v) for k, v in true_map.items()]))


def load_truth(path):
    import pandas as pd
    g = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {a: set(b.split(",")) if b else set() for a, b in zip(g.source1_entity_id, g.matched_entity_ids)}
