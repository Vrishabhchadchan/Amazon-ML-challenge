"""Glue shared by train.py and predict.py: load a split, attach context
features to all candidate pairs, then compute pair features for a subset."""
import os

import numpy as np
import pandas as pd

import config
from config import CACHE
from features import context_features, pair_features, record_extras


def load_split(split):
    s1, s2, s3 = [pd.read_parquet(os.path.join(CACHE, f"{split}_s{i}.parquet")) for i in (1, 2, 3)]
    others = pd.concat([s2, s3], ignore_index=True)
    pairs = pd.read_parquet(os.path.join(CACHE, f"{split}_pairs.parquet"))
    # one index space over S2 then S3
    pairs["o_idx"] = pairs.cand_idx.values + np.where(pairs.src.values == 3, len(s2), 0).astype(np.int32)
    pairs.drop(columns="cand_idx", inplace=True)
    return s1, others, pairs


def prune(pairs, ctx):
    """Cheap cut before the expensive features: keep a pair only if it ranks near
    the top of its S1 record on some channel. What survives is the final candidate
    set that the model scores (and what goes into candidate_pairs.tsv)."""
    keep = ((ctx.rank_both.values <= config.KEEP_RANK_BOTH)
            | (ctx.rank_name.values <= config.KEEP_RANK_ONE)
            | (ctx.rank_addr.values <= config.KEEP_RANK_ONE)
            | (ctx.c_rank_both.values <= 1))
    return keep


def build_features(s1, others, pairs, s1_rows=None, chunk=3_000_000, log=print):
    """Features for pruned pairs whose s1_idx is in s1_rows (all if None).
    Returns (keys, X) where keys has s1_idx/o_idx/src."""
    ctx = context_features(pairs)
    keep = prune(pairs, ctx)
    if s1_rows is not None:
        keep &= np.isin(pairs.s1_idx.values, s1_rows)
    pairs, ctx = pairs[keep], ctx[keep]
    x1, xo = record_extras(s1), record_extras(others)
    parts = []
    for i in range(0, len(pairs), chunk):
        p = pairs.iloc[i:i + chunk]
        parts.append(pd.concat([pair_features(p, s1, others, x1, xo), ctx.iloc[i:i + chunk]], axis=1))
        log(f"  features {min(i + chunk, len(pairs)):,}/{len(pairs):,}")
    X = pd.concat(parts)
    keys = pairs[["s1_idx", "o_idx", "src"]]
    return keys.reset_index(drop=True), X.reset_index(drop=True)


def decide(keys, prob, t_pair, t_top):
    """Final matching rule.
    A S1 entity gets matched at all only if its best candidate clears t_top
    (that guards singletons - a wrong match there costs a full 1.0), and then
    every candidate over t_pair is kept."""
    d = pd.DataFrame({"s1": keys.s1_idx.values, "o": keys.o_idx.values, "p": prob})
    best = d.groupby("s1").p.transform("max")
    d = d[(best >= t_top) & (d.p >= t_pair)]
    return d.groupby("s1").o.apply(list).to_dict()
