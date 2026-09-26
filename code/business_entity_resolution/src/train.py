"""Train the pairwise matcher and pick decision thresholds.

    python src/train.py [--frac 0.15]

Only a random slice of S1 entities is featurised (LightGBM doesn't need all
110M pairs), but blocking/context features were computed over the full train
split so their distribution matches test.
Out-of-fold predictions, grouped by S1 entity, are used to tune the thresholds
directly on macro F0.5 - the leaderboard metric, not pair-level F1.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config
from metrics import f05
from pipeline import build_features, decide, load_split

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=config.SEED, num_threads=os.cpu_count())


def labels(s1, others, keys):
    gt = pd.read_csv(os.path.join(config.DATA, "train", "train_ground_truth.tsv"), sep="\t",
                     dtype=str, keep_default_na=False)
    pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    poso = pd.Series(np.arange(len(others)), index=others.entity_id.values)
    e = gt[gt.matched_entity_ids != ""]
    e = e.assign(m=e.matched_entity_ids.str.split(",")).explode("m")
    true_keys = pos1[e.source1_entity_id.values].values.astype(np.int64) * 10**8 + poso[e.m.values].values
    truth = e.groupby(pos1[e.source1_entity_id.values].values).m.apply(
        lambda x: set(poso[x.values].values)).to_dict()
    k = keys.s1_idx.values.astype(np.int64) * 10**8 + keys.o_idx.values
    return np.isin(k, true_keys).astype(np.int8), truth


def score(pred, truth, rows):
    return float(np.mean([f05(set(pred.get(r, ())), truth.get(r, set())) for r in rows]))


def tune(keys, oof, truth, rows):
    best = (0, None)
    for t_top in np.arange(0.3, 0.95, 0.05):
        for t_pair in np.arange(0.2, 0.95, 0.05):
            if t_pair > t_top:
                continue
            s = score(decide(keys, oof, t_pair, t_top), truth, rows)
            if s > best[0]:
                best = (s, (round(float(t_pair), 2), round(float(t_top), 2)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()

    s1, others, pairs = load_split("train")
    rng = np.random.default_rng(config.SEED)
    rows = np.flatnonzero(rng.random(len(s1)) < args.frac)
    keys, X = build_features(s1, others, pairs, s1_rows=rows)
    del pairs
    y, truth = labels(s1, others, keys)
    n_true = sum(len(truth.get(r, ())) for r in rows)
    print(f"{len(rows):,} S1 entities, {len(X):,} pairs, {y.sum():,} positives, "
          f"blocking recall {y.sum() / n_true:.4f}  ({time.time() - t0:.0f}s)")

    oof = np.zeros(len(X), dtype=np.float32)
    iters = []
    for k, (tr, va) in enumerate(GroupKFold(args.folds).split(X, y, keys.s1_idx.values)):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  fold {k}: best_iter {m.best_iteration}")

    ceiling = score({r: list(truth.get(r, ())) for r in rows}, truth, rows)
    f, (t_pair, t_top) = tune(keys, oof, truth, rows)
    naive = score(decide(keys, oof, 0.5, 0.5), truth, rows)
    print(f"OOF macro F0.5: {f:.4f} at t_pair={t_pair} t_top={t_top} (0.5/0.5 gives {naive:.4f})")

    n_iter = int(np.mean(iters) * 1.1)
    model = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=n_iter)
    os.makedirs(config.CACHE, exist_ok=True)
    model.save_model(os.path.join(config.CACHE, "lgb.txt"))
    with open(os.path.join(config.CACHE, "thresholds.json"), "w") as fh:
        json.dump({"t_pair": t_pair, "t_top": t_top, "oof_f05": f, "features": list(X.columns)}, fh, indent=1)
    imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print(imp.head(15).round(0).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
