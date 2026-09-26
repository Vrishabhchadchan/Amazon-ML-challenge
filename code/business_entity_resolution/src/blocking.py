"""Candidate generation.

Each record becomes two sparse TF-IDF vectors over hashed tokens:
  name channel    - core name words, their phonetic squash, the space-less
                    concatenation (catches "evokeglass" vs "evoke glass")
  address channel - numbers, words, and within-segment bigrams

We stack them with equal weight, so the dot product is the mean of name and
address cosine, and take the top-K per source (S2 and S3 separately) for
every S1 record, within the same country. On top of that, name-only and
address-only passes catch records where one side is missing or mangled.
Very common tokens (df > MAX_DF) are dropped - they cost a lot of multiply
time and say almost nothing.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

import config

_hv = HashingVectorizer(analyzer=str.split, n_features=2**23, alternate_sign=False,
                        norm=None, binary=True, dtype=np.float32)


def name_doc(core, core_sq):
    toks = core.split()
    words = [t for t in toks if len(t) > 1]
    out = ["n" + t for t in words]
    out += ["q" + t for t in core_sq.split() if len(t) > 1]
    joined = "".join(toks)
    if len(toks) > 1:
        out.append("c" + joined)
    # char trigrams survive typos like "trrny"/"therapy"
    out += ["g" + joined[i:i + 3] for i in range(len(joined) - 2)]
    return " ".join(out)


def addr_doc(addr_seg):
    out = []
    for seg in addr_seg.split("|"):
        toks = seg.split()
        out += ["a" + t for t in toks]
        out += ["b" + a + "_" + b for a, b in zip(toks, toks[1:])]
    return " ".join(out)


def _idf_matrix(docs, max_df):
    X = _hv.transform(docs).tocsr()
    df = np.bincount(X.indices, minlength=X.shape[1])
    n = X.shape[0]
    idf = np.log((n + 1) / (df + 1)).astype(np.float32)
    # singletons can't match anything, very frequent tokens aren't worth the cost
    idf[(df < 2) | (df > max_df)] = 0
    X = X @ sp.diags(idf)
    X.eliminate_zeros()
    return l2norm(X).astype(np.float32).tocsr()


def vectorize(frames, max_df=config.MAX_DF):
    """frames: list of dataframes (same country). Returns (name, addr, both) matrices per frame."""
    sizes = [len(f) for f in frames]
    allf = pd.concat(frames, ignore_index=True)
    nd = [name_doc(c, q) for c, q in zip(allf.core, allf.core_sq)]
    ad = [addr_doc(s) for s in allf.addr_seg]
    Xn = _idf_matrix(nd, max_df)
    Xa = _idf_matrix(ad, max_df)
    w = np.float32(np.sqrt(0.5))
    Xb = sp.hstack([Xn * w, Xa * w], format="csr")
    out, start = [], 0
    for s in sizes:
        out.append((Xn[start:start + s], Xa[start:start + s], Xb[start:start + s]))
        start += s
    return out


def _topn(Q, I, k, thresh, chunk=200_000):
    It = I.T.tocsr()
    parts = []
    for i in range(0, Q.shape[0], chunk):
        parts.append(sp_matmul_topn(Q[i:i + chunk], It, top_n=k, threshold=thresh,
                                    n_threads=-1, sort=True))
    return sp.vstack(parts).tocoo()


def _rowdot(Q, X, r, c, chunk=500_000):
    """exact cosine for each (r[i], c[i]) pair - rows are already l2-normalised."""
    out = np.empty(len(r), dtype=np.float32)
    for i in range(0, len(r), chunk):
        a, b = Q[r[i:i + chunk]], X[c[i:i + chunk]]
        out[i:i + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def generate(s1, s2, s3, k=config.TOPK_PER_SOURCE, k_name=10, k_addr=10, query_mask=None, log=print):
    """Returns a DataFrame of (s1_idx, cand_idx, src, cos_name, cos_addr, cos_both)
    where cand_idx is the row position in s2 or s3 depending on src.

    query_mask: optional boolean array over s1 rows - only these are queried
    (the index side is always the full source, so the candidate density matches
    what the model will see at test time)."""
    res = []
    for country in sorted(s1.country.unique()):
        m1 = (s1.country == country).values
        if query_mask is not None:
            m1 &= query_mask
        if not m1.any():
            continue
        q = s1[s1.country == country]
        f2 = s2[s2.country == country]
        f3 = s3[s3.country == country]
        (Qn, Qa, Qb), (N2, A2, B2), (N3, A3, B3) = vectorize([q, f2, f3])
        sel = m1[(s1.country == country).values]
        Qn, Qa, Qb = Qn[sel], Qa[sel], Qb[sel]
        qidx = np.flatnonzero(m1)
        for src, (N, A, B, f) in {2: (N2, A2, B2, f2), 3: (N3, A3, B3, f3)}.items():
            if len(f) == 0:
                continue
            pos = f.index.values  # row positions in the full source frame
            keys = []
            for Qx, X, kk, th in [(Qb, B, k, 0.02), (Qn, N, k_name, 0.25), (Qa, A, k_addr, 0.25)]:
                C = _topn(Qx, X, kk, th)
                keys.append(C.row.astype(np.int64) * len(f) + C.col)
            keys = np.unique(np.concatenate(keys))
            r, c = keys // len(f), keys % len(f)
            res.append(pd.DataFrame({
                "s1_idx": qidx[r].astype(np.int32), "cand_idx": pos[c].astype(np.int32),
                "src": np.int8(src),
                "cos_name": _rowdot(Qn, N, r, c),
                "cos_addr": _rowdot(Qa, A, r, c),
                "cos_both": _rowdot(Qb, B, r, c),
            }))
            log(f"  {country} S{src}: {len(keys):,} pairs")
    return pd.concat(res, ignore_index=True)
