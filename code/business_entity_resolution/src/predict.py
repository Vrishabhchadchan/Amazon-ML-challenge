"""Score the test candidates and write the two submission files.

    python src/predict.py [--split test]

Features are built in slices of S1 rows so memory stays bounded; context
features are computed once over all pairs first (they need the full picture).
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from features import context_features, pair_features, record_extras
from pipeline import decide, load_split, prune


def write_lists(path, s1_ids, col, lists):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for i, sid in enumerate(s1_ids):
            fh.write(sid + "\t" + ",".join(lists.get(i, ())) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", type=int, default=150_000, help="S1 rows per feature batch")
    args = ap.parse_args()
    t0 = time.time()

    with open(os.path.join(config.CACHE, "thresholds.json")) as fh:
        th = json.load(fh)
    model = lgb.Booster(model_file=os.path.join(config.CACHE, "lgb.txt"))

    s1, others, pairs = load_split(args.split)
    ctx = context_features(pairs)
    keep = prune(pairs, ctx)
    pairs, ctx = pairs[keep].reset_index(drop=True), ctx[keep].reset_index(drop=True)
    print(f"{len(pairs):,} candidate pairs after pruning ({len(pairs) / len(s1):.1f}/query)")

    order = np.argsort(pairs.s1_idx.values, kind="stable")
    pairs, ctx = pairs.iloc[order].reset_index(drop=True), ctx.iloc[order].reset_index(drop=True)
    bounds = np.searchsorted(pairs.s1_idx.values, np.arange(0, len(s1) + args.slice, args.slice))

    x1, xo = record_extras(s1), record_extras(others)
    prob = np.zeros(len(pairs), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        p = pairs.iloc[a:b]
        X = pd.concat([pair_features(p, s1, others, x1, xo), ctx.iloc[a:b]], axis=1)
        prob[a:b] = model.predict(X[th["features"]])
        print(f"  scored {b:,}/{len(pairs):,}  ({time.time() - t0:.0f}s)")

    ids = others.entity_id.values
    keys = pairs[["s1_idx", "o_idx"]]
    matches = decide(keys, prob, th["t_pair"], th["t_top"])
    matches = {k: [ids[o] for o in v] for k, v in matches.items()}
    cands = pairs.groupby("s1_idx").o_idx.apply(lambda x: [ids[o] for o in x]).to_dict()

    for k, v in matches.items():  # a match must always come from the candidate set
        assert set(v) <= set(cands[k]), k

    os.makedirs(config.OUTPUT, exist_ok=True)
    s1_ids = s1.entity_id.values
    write_lists(os.path.join(config.OUTPUT, "matching_results.tsv"), s1_ids, "matched_entity_ids", matches)
    write_lists(os.path.join(config.OUTPUT, "candidate_pairs.tsv"), s1_ids, "candidate_entity_ids", cands)
    n = np.array([len(matches.get(i, ())) for i in range(len(s1))])
    print(f"wrote {len(s1):,} rows; {np.mean(n == 0):.3f} predicted singletons, "
          f"{n.mean():.2f} matches/entity  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
