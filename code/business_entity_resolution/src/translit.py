"""Learn a native-script -> English word dictionary from the training matches.

Plenty of S2/S3 names are Hindi/Tamil/Telugu/... renderings of the English S1
name ("பிரைவேட் லிமிடெட்" = "private limited"). unidecode alone gets these
only roughly right, so we align words in matched pairs and count how often
each native word lines up with each English word.

Alignment is dumb on purpose: split both sides on commas, and where segment
counts agree, split each segment on spaces and pair words by position when
the word counts agree. Noise washes out through the counting thresholds.
A word must be seen in >= MIN_ENT distinct S1 entities, so we only learn the
generic vocabulary (private, limited, builders, state names...) and not
per-business proper nouns - that keeps validation honest.

    python src/translit.py
"""
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from config import CACHE, DATA

MIN_ENT = 5
MIN_SHARE = 0.6
_native = re.compile(r"[ऀ-෿]")
_strip = re.compile(r"[!-/:-@[-`{-~]")  # ascii punctuation only - \w would eat indic vowel signs


def _words(s):
    return _strip.sub(" ", s).lower().split()


def _pairs(a, b):
    """yield (native_word, english_word) from two raw strings."""
    sa, sb = a.split(","), b.split(",")
    if len(sa) != len(sb):
        sa, sb = [a], [b]
    for x, y in zip(sa, sb):
        wx, wy = _words(x), _words(y)
        if len(wx) != len(wy):
            continue
        for u, v in zip(wx, wy):
            if _native.search(u) and not _native.search(v):
                yield u, v


def build():
    rd = lambda f: pd.read_csv(os.path.join(DATA, "train", f), sep="\t", dtype=str,
                               keep_default_na=False, quoting=3)
    s1 = rd("train_source1.tsv").set_index("entity_id")
    others = pd.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]).set_index("entity_id")
    gt = rd("train_ground_truth.tsv")
    gt = gt[gt.matched_entity_ids != ""]
    e = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")

    o = others.loc[e.m.values]
    has_native = (o.business_name.str.contains(_native) | o.business_address.str.contains(_native)).values
    e, o = e[has_native], o[has_native]
    a = s1.loc[e.source1_entity_id.values]

    seen = defaultdict(set)
    votes = defaultdict(Counter)
    for sid, n1, a1, n2, a2 in zip(e.source1_entity_id.values, a.business_name.values,
                                   a.business_address.values, o.business_name.values,
                                   o.business_address.values):
        for u, v in list(_pairs(n2, n1)) + list(_pairs(a2, a1)):
            votes[u][v] += 1
            seen[u].add(sid)

    vocab = {}
    for u, cnt in votes.items():
        if len(seen[u]) < MIN_ENT:
            continue
        v, c = cnt.most_common(1)[0]
        if c / sum(cnt.values()) >= MIN_SHARE:
            vocab[u] = v
    return vocab


if __name__ == "__main__":
    vocab = build()
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, "translit.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)
    print(len(vocab), "words learned")
    for k in list(vocab)[:30]:
        print(" ", k, "->", vocab[k])
