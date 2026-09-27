# Business Entity Resolution

Blocking (sparse TF-IDF top-K) → pair features (rapidfuzz + context ranks) → LightGBM →
threshold rule tuned on macro F0.5. Uses only the provided training data: no external
lookups, APIs or pretrained models.

## Setup

```
pip install -r requirements.txt
```

The default layout expects this folder at `student_resource/code/business_entity_resolution/`,
with the data in `student_resource/dataset/`. Override the locations with the `ER_DATA`,
`ER_CACHE` and `ER_OUTPUT` environment variables if needed.

## Run

```
python src/run_all.py
```

or step by step:

| step | command | what it does |
|---|---|---|
| 1 | `python src/translit.py` | learns native-script → English word map from train matches |
| 2 | `python src/prep.py --split train` / `--split test` | normalises names/addresses, caches parquet |
| 3 | `python src/candidates.py --split train` / `--split test` | blocking, within country |
| 4 | `python src/train.py` | LightGBM with grouped CV, tunes thresholds on OOF macro F0.5 |
| 5 | `python src/predict.py` | writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` |

Then validate:

```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Runtime

Measured on a 12-thread laptop with 16 GB RAM (no GPU needed):

| step | time |
|---|---|
| prep (per split) | ~2 min |
| candidates (per split) | ~80-90 min |
| train | ~50 min |
| predict | ~50 min |

Peak memory is about 9 GB during training, so don't run blocking and training at the same time.
`src/errors.py` optionally re-scores held-out train entities and prints example mistakes.
