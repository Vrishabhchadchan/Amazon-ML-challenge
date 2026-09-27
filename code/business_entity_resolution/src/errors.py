"""Score the final model on train entities it never saw and dump its mistakes.

train.py fits on a 15% slice of S1 entities (same seed), so everything outside
that slice is genuinely held out for the final model.

    python src/errors.py [--n 30000]
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from metrics import f05
from pipeline import build_features, decide, load_split
from train import labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--frac", type=float, default=0.15, help="must match train.py")
    args = ap.parse_args()

    with open(os.path.join(config.CACHE, "thresholds.json")) as fh:
        th = json.load(fh)
    model = lgb.Booster(model_file=os.path.join(config.CACHE, "lgb.txt"))

    s1, others, pairs = load_split("train")
    rng = np.random.default_rng(config.SEED)
    seen = rng.random(len(s1)) < args.frac  # identical draw to train.py
    unseen = np.flatnonzero(~seen)
    rows = np.random.default_rng(1).choice(unseen, args.n, replace=False)

    keys, X = build_features(s1, others, pairs, s1_rows=rows)
    del pairs
    y, truth = labels(s1, others, keys)
    prob = model.predict(X[th["features"]])
    pred = decide(keys, prob, th["t_pair"], th["t_top"])

    scores = np.array([f05(set(pred.get(r, ())), truth.get(r, set())) for r in rows])
    ctry = s1.country.values[rows]
    print(f"held-out macro F0.5 on {len(rows):,} unseen entities: {scores.mean():.4f}")
    for c in sorted(set(ctry)):
        print(f"  {c}: {scores[ctry == c].mean():.4f}")

    tp = fp = fn = 0
    fps, fns = [], []
    single_fp = 0
    for r in rows:
        p, t = set(pred.get(r, ())), truth.get(r, set())
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
        if not t and p:
            single_fp += 1
        fps += [(r, o) for o in p - t]
        fns += [(r, o) for o in t - p]
    print(f"pairs: precision {tp / (tp + fp):.4f}  recall {tp / (tp + fn):.4f}  "
          f"singletons wrongly matched: {single_fp}")

    cand = set(zip(keys.s1_idx.values, keys.o_idx.values))
    fn_blocked = sum((r, o) not in cand for r, o in fns)
    print(f"missed matches: {len(fns):,} ({fn_blocked:,} never reached the model - blocking)")

    def show(title, items, k=15):
        print(f"\n--- {title} ---")
        pick = np.random.default_rng(2).permutation(len(items))[:k]
        for i in pick:
            r, o = items[i]
            a, b = s1.iloc[r], others.iloc[o]
            print(f"{a.business_name} | {a.business_address}\n   -> {b.business_name} | {b.business_address}")

    show("false positives (wrong merges)", fps)
    show("false negatives (missed)", fns)


if __name__ == "__main__":
    main()
