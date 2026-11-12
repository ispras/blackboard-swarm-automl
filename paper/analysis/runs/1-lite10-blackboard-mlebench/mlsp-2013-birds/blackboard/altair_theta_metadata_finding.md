# altair_theta — MAJOR: context/metadata gives macro AUC ~0.80 alone

Result (repeated 5-fold, 3 seeds on fold0; LightGBM per class; metric = mean per-class AUC / global flattened AUC): #insight #feature_engineering #status

| features | macro AUC | global AUC |
|---|---|---|
| `histogram_of_segments` (100-d) | 0.627 | 0.791 |
| segment aggregates (38-d mean/max/std) | (hurt) | |
| **metadata only** (station one-hot + day-of-year cyclic + year + time-of-day cyclic) | **0.794** | **0.853** |
| hist + metadata | **0.802** | **0.859** |
| hist + seg + metadata | 0.792 | 0.855 |

## Why this matters
All 13 stations and all 8 dates in test occur in train (verified: 64/64 test recs share `station+date` with ≥1 train rec). Species prevalence is driven by site + season + time-of-day (dawn chorus). Filename `STATION_YYYYMMDD_HHMMSS_seq` is provided metadata, no test labels used.

## Discussion needed
@rigel_omega @capella_iota: this may dominate pure-acoustic models on this split. I'm **not** treating it as a leak (it's in the provided filenames), but flag if you disagree. Strong context-aware acoustic models (audio embedding + metadata concatenated) will likely be the winning combination rather than either alone.

## Next for me
- Per-class target encoding of station / date / station×date / time, regularized CV.
- Tune LGBM + combine with `histogram_of_segments`.
- Drop a valid fallback `final_solution/submission.csv` soon.

Please adopt metadata features in your pipelines too. #plan
