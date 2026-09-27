# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Pentium Predators  
**Team Members:** Vedant Kolhapure, Vrishabh Chadchan, Shrivardhan Kanaki, Harshadsai Badagu  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We use a classic two-stage entity-resolution pipeline: a sparse TF-IDF blocking stage that retrieves ~31 candidates per Source 1 record with 97.2% pair recall, followed by a LightGBM pairwise classifier over ~45 string-similarity and ranking features. The final match lists come from a two-threshold rule tuned directly on the leaderboard metric (macro F0.5). The key additions on top of that baseline are (a) a native-script → English word dictionary learned purely from the training matches, which handles Hindi/Tamil/Telugu/Bengali/... renderings of names and addresses, and (b) "competition" features that ask not only *how similar* a pair is but whether the candidate's best S1 match is this S1 record. On held-out training entities the pipeline scores **0.9655 macro F0.5**. No external data, APIs or pretrained models are used.

---

## 2. Methodology

### 2.1 Problem Analysis

Data size and shape:

| | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| Train | 2.21M (US 60%, India 40%) | 5.03M | 5.29M |
| Test | 1.73M (India 47%, US 38%, France 15%) | 4.89M | 5.08M |

Findings from the ground truth that shaped the design:

- **5.6% of S1 entities are singletons.** A wrong match on them costs a full 1.0, so the decision rule needs an explicit "is there any match at all" gate.
- **3.46 matches per entity on average** (usually 2–6), often several from the *same* source. One-to-one assignment would be wrong.
- **~26% of S2/S3 records match nothing** in S1, so the matcher has to reject a large pool of distractors.
- **Every true match shares the S1 record's country label** (100% in train). Blocking is done within country, and country is treated as an open set of labels, so France needs no special casing.
- No S2/S3 record is matched to two S1 entities. This motivates the candidate-side ranking features (Section 4).

Noise patterns observed in matched groups:

- **Native scripts.** Whole names or parts of them in Devanagari, Telugu, Tamil, Kannada, Bengali, Odia and Gujarati (`బాబా కేర్ ప్రైవేట్ లిమిటెడ్` ↔ `Baba Care Private Limited`), including mixed-script names (`Baba केयर Private Limited`). State names in addresses are also often native (`ఆంధ్రప్రదేశ్`).
- **Character noise.** Leetspeak substitutions (`Agricu1tural`), typos (`Couirrs`, `Trrny`), junk prefixes (`--`, `>>`, `M/s`, `Mr`), brackets (`[Box]`, `(Services)`), domain-style names (`EVOKEGLASSCOM`, `wilfordhancock.com`).
- **Legal and trade-name variation.** `Pvt Ltd` / `Private Limited` / `प्रा. लि.`, dropped or added `Inc`, `LLC`, `Center`, `Group`.
- **Addresses.** Comma-separated components shuffled in any order; states as full names or codes (`Delaware`/`DE`, `Maharashtra`/`MH`); house numbers perturbed (`0407` vs `407`, `6238b`, `Fourth` vs `4th`, `7-153` vs `7/153`); components dropped (city, state, unit); placeholder tokens (`NULL`, `<NULL>`); 3% of S2/S3 addresses empty.
- **Unrelated names at a shared address.** Some true matches carry a completely different (generated) name at the same address, so address evidence alone has to be able to carry a match.
- **France (test only).** We read the French test records, without labels, to check the normalisation. Findings: `R`/`R.`/`AV`/`BD`/`ALL`/`ESPL` street abbreviations, `ST`/`STE` = Saint/Sainte (not "street"), `N°` house-number markers, elided articles (`D'ESPAGNE`), `BIS`/`TER` suffixes, legal forms `SARL/SAS/SASU/EURL/SCI/EI`. Postcodes are almost never present (0.5%).

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier (LightGBM) + metric-tuned decision rule.  
**Core Innovation:**
1. A **learned native-script dictionary**: word alignment over the training matches, used only for vocabulary seen across ≥5 distinct entities, so it captures generic words (private, limited, builders, state names) rather than memorising businesses.
2. **Competition (context) features**: for every pair we compute how the candidate ranks among *all* S1 records that retrieved it, and how the S1 record ranks among its own candidates. This turned out to be the single most important signal (Section 5).
3. **Two-threshold decision rule tuned on macro F0.5**, which protects singletons.

Pipeline:

```
raw TSV ─► normalise (names, addresses, native script, numbers, per-country rules)
        ─► blocking: hashed TF-IDF, 3 channels, top-K per source, within country   (~52 cands / S1)
        ─► context features over all pairs  ─► prune to ~31 cands / S1  ─► candidate_pairs.tsv
        ─► pair features (rapidfuzz, numbers, legal forms)  ─► LightGBM probability
        ─► decision rule (t_top gate + t_pair cut)  ─► matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

**Normalisation first** (`normalize.py`). This matters as much as the blocker itself:
- Native-script words go through the learned dictionary, then `unidecode` romanises what remains.
- Names: lowercase, `&`→and, leetspeak repair inside alphabetic tokens, canonical legal forms (also matched phonetically, so romanised `praiveett` → `pvt`), honorific and junk-token removal, trailing `com` stripped from domain-style names. The *core name* is the name without legal forms.
- Addresses: split on commas into segments (order-free downstream), expand abbreviations, state names/codes → a tagged code (`st_de`), number canonicalisation (`0407`→`407`, `6238b`→`6238`, `fourth`→`4`, `7-153`→`7_153`), unit markers and `NULL` placeholders dropped.
- Country-specific rules are keyed on the country string and fall back to generic behaviour for any other label. French rules only fire for `France`, so they cannot affect US/India.

**Blocking keys** (`blocking.py`). Each record becomes two sparse vectors over hashed tokens (2²³ buckets, so no vocabulary is held in memory):
- *Name channel:* core-name words, their phonetic squash (vowels dropped, repeats collapsed), the space-less concatenation (`evokeglass`), and character trigrams of the concatenation (typo-robust).
- *Address channel:* address words, numbers, and within-segment bigrams (`fox_pointe`), which are rare and specific.

Weights are IDF computed within country. Tokens that occur only once, or more than 25,000 times, are zeroed: they either can't match or add a lot of multiply cost for almost no signal. Rows are L2-normalised.

**Retrieval.** Within each country and separately for S2 and S3, we take the union of three sparse top-K searches (`sparse_dot_topn`, multithreaded):
- combined name+address vector (equal weights), top 20
- name only, top 10: catches empty or garbled addresses
- address only, top 10: catches native-script or unrelated names at the right address

Exact name, address and combined cosines are then computed for every retrieved pair.

**Pruning.** Rank features are computed over all pairs. A pair is kept if it is in the top 10 on the combined score, the top 3 on name or address alone, or is the candidate's best S1 match. This set is exactly what the model scores and what is written to `candidate_pairs.tsv`.

| | Train | Test |
|---|---|---|
| Pairs after retrieval | 114.2M (51.7 / S1) | 89.6M (51.7 / S1) |
| Pairs after pruning (= candidate set) | ~31 / S1 | 54.0M (31.2 / S1) |
| Pair recall of the candidate set | **97.17%** | n/a |
| Reduction vs all within-country pairs | > 99.999% | > 99.999% |

**How we made sure true matches were not lost.** Recall was measured on a 3% probe of train queries after every change, and each fix targeted a class of misses found by reading them:
- word-level blocking only: 87.0%
- plus character trigrams and separate name-only/address-only passes: 94.6%
- plus the native-script dictionary: 95.2%
- plus raising the frequency cap and canonicalising numbers: **97.1%**

Pruning from ~52 to ~31 candidates cost less than 0.05 points of recall.

---

## 4. Matching Model

**Features used** (~45 per pair, `features.py`):
- **Name features:** rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` and Jaro–Winkler on the core name; ratio on the space-less name; token-set ratio on the phonetic squash and on the full name including legal forms; first-token equality; acronym match (`IBM` ↔ initials of the other name); token counts and their difference; legal-form agreement and legal-form clash.
- **Address features:** token-set, token-sort and partial token-set ratios on the normalised address; address length ratio; empty-address flag; number-set Jaccard and a "both have numbers but share none" clash flag; leading house-number equality and edit similarity.
- **Blocking scores:** exact TF-IDF cosine for the name, address and combined channels.
- **Context / competition features** (over *all* pairs of the split, before pruning):
  - Per S1 record and source: rank and gap-to-best on combined, name and address cosine; number of candidates; number of strong candidates.
  - Per candidate: rank of this S1 record among all S1 records that retrieved the candidate (`c_rank_both`), gap to the candidate's best S1 score, rank on name, and how many S1 records retrieved it.
- **Other:** source (S2/S3) and a native-script flag on the candidate. Country is deliberately *not* a feature, so the model transfers to France.

**Model type:** LightGBM binary classifier (MIT licence; ~2k trees, far below the 8B-parameter limit). Settings: learning rate 0.05, 127 leaves, `min_data_in_leaf` 100, 0.8 feature/bagging fractions, L2 1.0. Training uses a 15% random sample of train S1 entities (330,664 entities, 10.4M pairs, 1.11M positives). Blocking and context features are computed on the full train split so their distributions match test. Evaluation uses 4-fold `GroupKFold` grouped by S1 entity, so no entity is split across folds. The final model is refit on the whole sample with 1.1× the mean best iteration.

**Threshold selection method:** grid search on out-of-fold probabilities, directly maximising the macro F0.5 of the challenge (per S1 entity, singletons included, blocking misses counted as recall loss). The decision rule:
- if the best candidate of an S1 record has p < **t_top = 0.75**, predict no match (singleton gate);
- otherwise output every candidate with p ≥ **t_pair = 0.70**.

Tuning moved F0.5 from 0.9614 (0.5/0.5) to 0.9655. Both thresholds end up well above 0.5, as expected for a precision-weighted metric.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), 4-fold out-of-fold on 330k train entities:** **0.9655**
- **Held-out check with the final model** (30,000 train entities outside its training slice): **0.9656** (India 0.9604, US 0.9691). This matches the CV score, so the model is not overfitting its training slice.

On those 30k held-out entities:

| | Value |
|---|---|
| Pair precision | 0.9897 |
| Pair recall | 0.9277 |
| Singletons wrongly matched | 77 of ~1,700 (≈4.5%) |
| Missed true matches | 7,522, of which 2,970 (39%) were lost in blocking and 4,552 rejected by the model |

Precision is very high and recall is the main loss, which is the trade-off F0.5 asks for.

- **Common false positives (wrong merges):**
  - *Sibling businesses at the same address with near-identical names*: `MS Energy Pvt Ltd` ↔ `ZS Energy Private`, `Kbs Manufacturing` ↔ `Kbs Software` (door 6-195/1 vs 6-195/10), `Stupendous Development` ↔ `Stupendous Projects`. Name and address both score very high, and only a single letter or house-number digit differs.
  - *Same name, empty address*: `Dental Trusted Associates`, `TD Electronic`, `Innovative Lease Inc`. With no address the model cannot tell a genuine duplicate from a same-named different business, and the ground truth says "different" here.
  - *Unrelated generated name at an exact address*: `lgrill.com`, `colonialmedicalcenter.com` at the S1 record's exact street address. These match the "unrelated name, shared address" pattern that is sometimes a true match, so a few slip through.
- **Common false negatives (missed matches):**
  - *Same name, empty address*: the mirror image of the false-positive case (`Express National Textiles Corp`, `Kolkata Capital Private Limited`, `Cunningham Dynamix`). The model is deliberately cautious here because F0.5 punishes the wrong merges above more than it rewards these.
  - *Unrelated name + partial address*: `Orbiaria`, `Vantagepyra` at a matching but reordered/partial address. There is no name signal, so the address alone has to clear a high threshold.
  - *Native-script names with truncated addresses*: `વન એન્જિનિયરિંગ પ્રાઇવેટ લિમિટેડ` ↔ `One Engineering Private Limited`. The dictionary covers generic words, but rarer name words fall back to rough romanisation.
  - *Blocking misses (39% of all misses)*: mostly records where both name and address are heavily degraded or empty. They never reach the model.


Feature importance (gain, top 10): `c_rank_both`, `cos_both`, `c_gap_both`, `a_tset`, `num_jac`, `c_n_s1`, `lead_sim`, `num_clash`, `n_full_tset`, `legal_clash`. The competition features dominate. Near-duplicate businesses (same chain, same building) are common, and "is this the candidate's best S1 match" separates them better than any absolute similarity. House-number agreement is the strongest pure address signal.

Test-set sanity check (no labels): predicted singleton rate 5.8% (train truth 5.6%), 3.31 matches per entity (train truth 3.46). By country: France 5.0% / 3.45, India 5.9% / 3.27, US 5.9% / 3.30. France, which has no training data, behaves like the seen countries.

---

## 6. Conclusion

A carefully normalised, recall-oriented blocker plus a gradient-boosted matcher with competition-aware features reaches 0.9655 macro F0.5 using only the provided data. The largest gains came from reading the misses, not from model capacity: native-script handling, number canonicalisation and not over-pruning frequent tokens each moved blocking recall by whole points, and the candidate-side ranking features were the model's strongest signal. The remaining headroom is mostly the ~2.8% of true matches that never reach the model. Better recall on heavily abbreviated or native-only records would be the next step.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| File | Role |
|---|---|
| `src/config.py` | paths (overridable via `ER_DATA`, `ER_CACHE`, `ER_OUTPUT`) and blocking/pruning constants |
| `src/normalize.py` | name/address normalisation, per-country rules |
| `src/translit.py` | learns the native-script dictionary from train matches → `cache/translit.json` |
| `src/prep.py` | normalises all six source files once, caches parquet (multiprocessing) |
| `src/blocking.py`, `src/candidates.py` | hashed TF-IDF blocking → `cache/{split}_pairs.parquet` |
| `src/features.py`, `src/pipeline.py` | context features, pruning, pair features, decision rule |
| `src/train.py` | LightGBM with grouped CV, threshold tuning → `cache/lgb.txt`, `cache/thresholds.json` |
| `src/predict.py` | scores test candidates, writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` |
| `src/metrics.py` | macro F0.5 exactly as defined by the challenge |
| `src/errors.py` | held-out evaluation and error dump used for Section 5 |
| `src/run_all.py` | entry point: runs every step in order |

Reproduce end to end with `pip install -r requirements.txt` then `python src/run_all.py`. Runtime on a 12-thread laptop with 16 GB RAM: prep ~2 min per split, blocking ~80–90 min per split, training ~50 min, prediction ~50 min. No GPU needed. Random seed 42 throughout.

### B. Additional Results

Blocking recall progression (3% probe of train queries):

| Change | Pair recall | Candidates / S1 |
|---|---|---|
| Word tokens, combined channel, K=12 | 87.0% | 24.5 |
| + char trigrams, name-only & address-only passes, K=20 | 94.6% | 50.7 |
| + learned native-script dictionary | 95.2% | 50.7 |
| + max-df 3,000 → 25,000, number canonicalisation | 97.1% | 51.8 |
| + pruning (final candidate set, full train) | 97.2% | ~31 |

Cross-validation: best iterations per fold 2000 / 1999 / 1823 / 1937.
