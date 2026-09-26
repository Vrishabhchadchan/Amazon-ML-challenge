"""Read raw TSVs, normalise every record once, cache as parquet.

    python src/prep.py --split train
    python src/prep.py --split test
"""
import argparse
import os
from multiprocessing import Pool

import pandas as pd

from config import CACHE, DATA
from normalize import normalize_record


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def _norm_chunk(rows):
    return [normalize_record(n, a, c) for n, a, c in rows]


def normalize_frame(df, workers):
    rows = list(zip(df.business_name, df.business_address, df.country))
    step = 20000
    chunks = [rows[i:i + step] for i in range(0, len(rows), step)]
    with Pool(workers) as pool:
        out = []
        for part in pool.imap(_norm_chunk, chunks):
            out.extend(part)
    norm = pd.DataFrame(out, index=df.index)
    return pd.concat([df, norm], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    for src in (1, 2, 3):
        out = os.path.join(CACHE, f"{args.split}_s{src}.parquet")
        if os.path.exists(out):
            print("exists, skipping", out)
            continue
        df = read_tsv(os.path.join(DATA, args.split, f"{args.split}_source{src}.tsv"))
        df["country"] = df.country.str.strip()
        df = normalize_frame(df, args.workers)
        df.to_parquet(out, index=False)
        print(f"{args.split} source{src}: {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
