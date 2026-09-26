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

Hardware used: 12-core CPU, 16 GB RAM. No GPU needed.
