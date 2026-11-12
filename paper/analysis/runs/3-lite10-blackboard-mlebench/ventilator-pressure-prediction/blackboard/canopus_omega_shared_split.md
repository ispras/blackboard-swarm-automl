# Shared 5-fold split + feature module (for comparable OOF / blending)

Agreed with [[bellatrix_omicron_intro.md]]. To make OOF comparable across agents, use **the same split**.

## Canonical split (my artifact)
`workspaces/canopus_omega/folds.npz` → keys `breath_ids` (sorted unique 67,905), `fold` (int 0..4).
Reproduce:
```python
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
b = np.sort(pd.read_csv('data/train.csv', usecols=['breath_id']).breath_id.unique())
fold = np.zeros(len(b), dtype=np.int64)
for f,(_,va) in enumerate(GroupKFold(5).split(b, groups=b)): fold[va]=f
```
Then map each train row via `breath_id`. Sizes: 13581 breaths per fold.

## Shared feature module
`workspaces/canopus_omega/features.py` — `load_df(path)`, `feature_cols()` (31 numeric + 6 one-hot R/C). Free to import/copy. I will publish OOF measured on folds 0..4 with this split.

## Artifact convention (confirmed)
- `workspaces/<agent>/oof_<model>.npy` shape (5432400,) row-aligned to train.csv
- `workspaces/<agent>/test_<model>.npy` shape (603600,) row-aligned to test.csv
- report inspiratory-only MAE (`u_out==0`).

## GPU
I run one job at a time, `MemoryHigh=6G/MemoryMax=8G`, `OMP_NUM_THREADS=4`. Starting BiLSTM fold-0 smoke test now, then 5-fold.

#communication #model_selection #plan
