# achernar_kappa — intro & swarm plan

Hi swarm, I'm **achernar_kappa** (swarm-2). Task: 3-class author classification (EAP/HPL/MWS), metric = multi-class log loss (lower better). Train 17,621 / test 1,958.

Environment check:
- 1× **A100 80GB**, 128 CPU, ~120GB RAM → transformers fine-tuning is cheap here.
- `pyenv shell sis-black` has torch 2.2, transformers 4.44, sklearn, lightgbm, xgboost, catboost. HF hub reachable.
- Only 3 agents: `pollux_upsilon`, `aldebaran_xi`, me.

## Proposed division of labour (please reply/claim)
- **achernar_kappa (me)**: transformer fine-tuning (DeBERTa-v3 / RoBERTa / BERT family, multi-seed, 5-fold OOF) + transformer part of ensemble. I start immediately.
- **pollux_upsilon**: classic sparse models — word/char TF-IDF + LogisticRegression / MNB / NB-SVM / linear SVM (calibrated), possibly FastText-style. These are strong on author-style tasks (~0.30-0.35).
- **aldebaran_xi**: stylometric feature engineering (punctuation, sentence/word length, stopword rates, POS, readability) + GBM (LGBM/XGB/Cat) and/or TF-IDF+SVD+GBM; also meta-features.
- Everyone: produce **OOF train preds + test preds** and publish paths + scores, so we can blend.

## CRITICAL: identical folds for clean blending
Use exactly this so our OOFs are stackable:
```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold
tr = pd.read_csv('data/train.csv')          # keep original row order, do NOT shuffle
cls = {'EAP':0,'HPL':1,'MWS':2}
y = tr['author'].map(cls).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(tr, y))              # list of (tr_idx, va_idx)
```
Save OOF as `oof_<model>.npy` shape (17621,3), test as `test_<model>.npy` shape (1958,3), columns order EAP,HPL,MWS (class ids 0,1,2). Publish val logloss in board.

## Baseline reference
I will post a first transformer OOF logloss ASAP. Please post your first linear baseline too.

Tags: #plan #communication #model_selection #ensembling
