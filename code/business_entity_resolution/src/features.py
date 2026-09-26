"""Pair features for the matcher.

Three groups:
  * pairwise string similarity (rapidfuzz, run element-wise in C via cpdist)
  * address number agreement - house/plot numbers are the most discriminative
    thing in an address and also where the noise injector likes to poke
  * context - how this candidate ranks among the other candidates of the same
    S1 record, and how this S1 ranks among all S1 records that pulled in the
    same candidate. These matter a lot: near-duplicate businesses exist, and the
    right answer is usually the relatively best one, not just a high absolute score.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

W = -1  # rapidfuzz workers: all cores


def _lead_num(addr):
    for t in addr.split():
        if t[0].isdigit():
            return t
    return ""


def _initials(core):
    toks = core.split()
    return "".join(t[0] for t in toks) if len(toks) > 1 else ""


def record_extras(df):
    """Per-record helpers computed once, not once per pair."""
    return pd.DataFrame({
        "lead": [_lead_num(a) for a in df.addr_n.values],
        "ini": [_initials(c) for c in df.core.values],
        "ntok": df.core.str.count(" ").add(1).where(df.core != "", 0).astype(np.int16).values,
        "flat": df.core.str.replace(" ", "", regex=False).values,
    })


def _sim(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32) / 100.0


def _set_stats(a, b):
    """jaccard and conflict flag for space-separated token sets."""
    jac = np.zeros(len(a), dtype=np.float32)
    conflict = np.zeros(len(a), dtype=np.int8)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        sx, sy = set(x.split()), set(y.split())
        inter = len(sx & sy)
        jac[i] = inter / len(sx | sy)
        conflict[i] = inter == 0
    return jac, conflict


def pair_features(pairs, s1, others, x1, xo):
    """pairs: s1_idx, o_idx (row in `others`), src, cos_* columns.
    s1/others: normalised frames; x1/xo: record_extras of each."""
    i, j = pairs.s1_idx.values, pairs.o_idx.values
    A = s1.iloc[i]
    B = others.iloc[j]
    XA, XB = x1.iloc[i], xo.iloc[j]
    f = pd.DataFrame(index=pairs.index)
    for c in ("cos_name", "cos_addr", "cos_both"):
        f[c] = pairs[c].values
    f["src"] = pairs.src.values

    ca, cb = A.core.values, B.core.values
    f["n_ratio"] = _sim(ca, cb, fuzz.ratio)
    f["n_tset"] = _sim(ca, cb, fuzz.token_set_ratio)
    f["n_tsort"] = _sim(ca, cb, fuzz.token_sort_ratio)
    f["n_partial"] = _sim(ca, cb, fuzz.partial_ratio)
    f["n_jw"] = process.cpdist(ca, cb, scorer=JaroWinkler.normalized_similarity,
                               workers=W, dtype=np.float32)
    f["n_flat"] = _sim(XA.flat.values, XB.flat.values, fuzz.ratio)
    f["n_sq"] = _sim(A.core_sq.values, B.core_sq.values, fuzz.token_set_ratio)
    f["n_full_tset"] = _sim(A.name_n.values, B.name_n.values, fuzz.token_set_ratio)
    f["n_first_eq"] = (A.core.str.split(n=1).str[0].values == B.core.str.split(n=1).str[0].values).astype(np.int8)
    ia, ib = XA.ini.values, XB.ini.values
    f["acronym"] = (((ia != "") & (ia == XB.flat.values)) | ((ib != "") & (ib == XA.flat.values))).astype(np.int8)
    f["ntok_a"], f["ntok_b"] = XA.ntok.values, XB.ntok.values
    f["ntok_diff"] = np.abs(f.ntok_a - f.ntok_b)

    la, lb = A.legal.values, B.legal.values
    f["legal_eq"] = ((la == lb) & (la != "")).astype(np.int8)
    f["legal_clash"] = ((la != "") & (lb != "") & (la != lb)).astype(np.int8)

    aa, ab = A.addr_n.values, B.addr_n.values
    f["a_empty"] = (ab == "").astype(np.int8)
    f["a_tset"] = _sim(aa, ab, fuzz.token_set_ratio)
    f["a_tsort"] = _sim(aa, ab, fuzz.token_sort_ratio)
    f["a_partial"] = _sim(aa, ab, fuzz.partial_token_set_ratio)
    f["a_len_ratio"] = (np.char.str_len(ab.astype(str)) + 1) / (np.char.str_len(aa.astype(str)) + 1)

    f["num_jac"], f["num_clash"] = _set_stats(A.nums.values, B.nums.values)
    lda, ldb = XA.lead.values, XB.lead.values
    has = (lda != "") & (ldb != "")
    f["lead_eq"] = np.where(has, (lda == ldb).astype(np.int8), -1).astype(np.int8)
    f["lead_sim"] = np.where(has, _sim(lda, ldb, fuzz.ratio), -1).astype(np.float32)
    f["native_b"] = B.native.values.astype(np.int8)
    return f


def context_features(pairs):
    """Rank/margin features from the blocking cosines. Run on *all* pairs of a
    split so the candidate-side view (which S1 records compete for the same
    S2/S3 record) is complete."""
    g1 = pairs.s1_idx.values
    tmp = pd.DataFrame({"g1": g1, "gc": pairs.o_idx.values, "src": pairs.src.values,
                        "both": pairs.cos_both.values, "name": pairs.cos_name.values,
                        "addr": pairs.cos_addr.values})
    f = pd.DataFrame(index=pairs.index)
    by1 = tmp.groupby(["g1", "src"], sort=False)
    for c in ("both", "name", "addr"):
        f[f"rank_{c}"] = by1[c].rank(ascending=False, method="min").astype(np.float32).values
        f[f"gap_{c}"] = (tmp[c] - by1[c].transform("max")).astype(np.float32).values
    f["n_cand"] = tmp.groupby("g1", sort=False).both.transform("size").astype(np.float32).values
    tmp["strong"] = tmp.both > 0.7
    f["n_strong"] = by1.strong.transform("sum").astype(np.float32).values
    byc = tmp.groupby("gc", sort=False)
    f["c_rank_both"] = byc.both.rank(ascending=False, method="min").astype(np.float32).values
    f["c_gap_both"] = (tmp.both - byc.both.transform("max")).astype(np.float32).values
    f["c_rank_name"] = byc.name.rank(ascending=False, method="min").astype(np.float32).values
    f["c_n_s1"] = byc.both.transform("size").astype(np.float32).values
    return f
