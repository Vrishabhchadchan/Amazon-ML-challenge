"""Text cleanup for business names and addresses.

Everything here is plain string munging - no lookups against anything outside
the provided data. Indic scripts are romanised with unidecode, which is rough
but gets addresses/names close enough for char-level similarity to work.
"""
import json
import os
import re
from functools import lru_cache

from unidecode import unidecode

LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "llc": "llc",
    "llp": "llp", "lp": "lp", "plc": "plc", "pvt": "pvt", "private": "pvt",
    "pte": "pvt", "pl": "pvt", "sarl": "sarl", "sas": "sas", "sa": "sa",
    "eurl": "eurl", "sasu": "sas", "sci": "sci", "snc": "snc", "gmbh": "gmbh",
    "ltda": "ltd", "group": "group", "grp": "group", "holdings": "holdings",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent",
    "intl": "international", "international": "international",
    "svcs": "services", "services": "services", "service": "services",
    "mfg": "manufacturing", "bros": "brothers", "assoc": "associates",
    "associates": "associates", "trust": "trust",
}
# words that carry no identity at all - honorifics and scrape noise
FILLER = {"the", "and", "of", "m", "s", "ms", "mr", "mrs", "smt", "sri", "shri",
          "dr", "dba", "aka", "www", "com", "net", "org", "in", "fr", "null", "na"}
# legal-ish tokens we drop when building the "core" name
CORE_DROP = set(LEGAL.values()) - {"international", "services", "manufacturing",
                                   "brothers", "trust", "group"}

ADDR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace",
    "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nr": "near", "opp": "opposite", "bhd": "behind", "sec": "sector",
    "no": "number", "hno": "number", "dist": "district", "tq": "taluk",
    "tal": "taluk", "po": "postoffice", "ps": "policestation", "mkt": "market",
    "ngr": "nagar", "clny": "colony", "r": "rue", "bd": "boulevard",
    "imp": "impasse", "fbg": "faubourg", "rte": "route", "chem": "chemin",
    "pte": "porte", "cc": "centrecommercial", "zi": "zoneindustrielle",
    "za": "zoneactivite", "unit": "unit",
}
# drop unit markers entirely - "# unit 6", "unit unit 6" and "6" should agree
ADDR_DROP = {"unit", "number", "apartment", "suite", "floor", "the", "of", "null",
             "na", "c", "o", "co"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py",
}
# state names get replaced by a tagged code so "Delaware"/"DE" collide
_STATE_RE = {
    c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, m), key=len, reverse=True)) + r")\b")
    for c, m in (("us", US_STATES), ("india", IN_STATES))
}
_STATE_MAP = {"us": US_STATES, "india": IN_STATES}

ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
            "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_non_alnum = re.compile(r"[^a-z0-9 ]+")
_ws = re.compile(r"\s+")


_native_run = re.compile("[ऀ-෿‌‍]+")
_VOCAB = None


def _vocab():
    # learned by translit.py from the training matches; empty if not built yet
    global _VOCAB
    if _VOCAB is None:
        from config import CACHE
        path = os.path.join(CACHE, "translit.json")
        _VOCAB = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                _VOCAB = json.load(fh)
    return _VOCAB


def _num(t):
    """'0407' -> '407', '6238b' -> '6238', '4th' -> '4', '032_29' -> '32_29'."""
    t = re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", t)
    t = re.sub(r"^(\d+)[a-z]$", r"\1", t)
    return "_".join(p.lstrip("0") or "0" for p in t.split("_"))


def _ascii(s):
    s = s or ""
    if _native_run.search(s):
        v = _vocab()
        s = _native_run.sub(lambda m: " " + v.get(m.group(0), m.group(0)) + " ", s)
    return unidecode(s).lower()


def _fix_leet(tok):
    # "agricu1tural" -> "agricultural", but leave pure numbers alone
    if tok.isdigit() or not any(ch.isdigit() for ch in tok):
        return tok
    if sum(ch.isalpha() for ch in tok) < 2:
        return tok
    return tok.translate(_LEET)


@lru_cache(maxsize=None)
def squash(tok):
    """Crude phonetic key: drop vowels after the first char, collapse repeats.
    Helps transliterations: 'bilddrs' and 'builders' both -> 'bldrs'."""
    if not tok:
        return tok
    out = [tok[0]]
    for ch in tok[1:]:
        if ch in "aeiouyh":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


_LEGAL_SQ = None


def _legal_sq():
    # romanised Hindi/Telugu legal words: "praiveett" and "private" share "prvt"
    global _LEGAL_SQ
    if _LEGAL_SQ is None:
        _LEGAL_SQ = {squash(k): v for k, v in LEGAL.items() if len(k) >= 5}
    return _LEGAL_SQ


def name_tokens(raw):
    s = _ascii(raw).replace("&", " and ").replace("+", " and ")
    s = s.replace("'", "").replace(".", "")
    s = _non_alnum.sub(" ", s)
    toks = []
    for t in s.split():
        t = _fix_leet(t)
        if t in LEGAL:
            t = LEGAL[t]
        elif len(t) >= 5:
            t = _legal_sq().get(squash(t), t)
        if len(t) > 6 and t.endswith("com") and len(s.split()) == 1:
            t = t[:-3]
        if t in FILLER:
            continue
        toks.append(t)
    return toks


def core_name(toks):
    core = [t for t in toks if t not in CORE_DROP]
    return core or toks


def addr_tokens(raw, country):
    s = _ascii(raw).replace("#", " ")
    ckey = country.strip().lower()
    if ckey in _STATE_RE:
        m = _STATE_MAP[ckey]
        s = _STATE_RE[ckey].sub(lambda x: " st_" + m[x.group(1)] + " ", s)
    s = s.replace("'", "")
    # keep hyphen/slash numbers together: "7-153", "3/115"
    s = re.sub(r"(\d)\s*[-/]\s*(\d)", r"\1_\2", s)
    s = re.sub(r"[^a-z0-9_ ,]+", " ", s)
    segs = []
    for seg in s.split(","):
        parts = seg.split()
        if len(parts) == 1 and ckey in _STATE_MAP and parts[0] in _codes(ckey):
            segs.append(["st_" + parts[0]])
            continue
        out = []
        for t in parts:
            t = ORDINALS.get(t, t)
            if t[0].isdigit():
                t = _num(t)
            t = ADDR.get(t, t)
            if t in ADDR_DROP:
                continue
            out.append(t)
        if out:
            segs.append(out)
    return segs


@lru_cache(maxsize=None)
def _codes(ckey):
    return frozenset(_STATE_MAP[ckey].values())


def normalize_record(name, addr, country):
    ntoks = name_tokens(name)
    core = core_name(ntoks)
    segs = addr_tokens(addr, country)
    atoks = []
    for seg in segs:
        atoks.extend(seg)
    nums = sorted({t for t in atoks if t[0].isdigit()})
    return {
        "name_n": " ".join(ntoks),
        "core": " ".join(core),
        "core_sq": " ".join(squash(t) for t in core if not t.isdigit()),
        "legal": " ".join(sorted(set(ntoks) - set(core))),
        "addr_n": " ".join(atoks),
        "addr_seg": "|".join(" ".join(seg) for seg in segs),
        "nums": " ".join(nums),
        "native": int(any(ord(ch) > 0x900 for ch in (name or ""))),
    }
