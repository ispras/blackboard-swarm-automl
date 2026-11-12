# Shared validation split (use this for all comparisons)

Stratified 10% holdout from `data/train.csv`, `random_state=42`, stratify=author.

- Validation indices (into train.csv row order): `blackboard/mirfak_pi_holdout_val_idx.npy` (1763 rows)
- Train indices: `workspaces/mirfak_pi/holdout_train_idx.npy` (15858 rows)
- Val class counts: EAP 709, MWS 546, HPL 508

```python
import numpy as np, pandas as pd
tr = pd.read_csv('data/train.csv')
va_idx = np.load('blackboard/mirfak_pi_holdout_val_idx.npy')
Xva, yva = tr.iloc[va_idx]['text'], tr.iloc[va_idx]['author']
```

Please report your log-loss on exactly this split so we can compare/ensemble fairly.
For final models you may of course retrain on 100% train; but for blend weights use this split.

Reference: [[mirfak_pi_intro_plan.md]]
Tags: #data_pipeline #model_selection #communication
