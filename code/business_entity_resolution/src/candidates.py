"""Run blocking for every S1 record of a split and cache the pairs.

    python src/candidates.py --split train
"""
import argparse
import os
import time

import pandas as pd

import blocking
from config import CACHE


def load(split):
    return [pd.read_parquet(os.path.join(CACHE, f"{split}_s{i}.parquet")) for i in (1, 2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    t = time.time()
    s1, s2, s3 = load(args.split)
    pairs = blocking.generate(s1, s2, s3, log=lambda m: print(m, flush=True))
    out = os.path.join(CACHE, f"{args.split}_pairs.parquet")
    pairs.to_parquet(out, index=False)
    print(f"{len(pairs):,} pairs for {len(s1):,} queries ({len(pairs) / len(s1):.1f}/query) "
          f"in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
