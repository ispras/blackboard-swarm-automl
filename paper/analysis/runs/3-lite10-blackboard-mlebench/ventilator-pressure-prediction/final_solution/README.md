# Google Brain — Ventilator Pressure Prediction — swarm solution

Task: predict `pressure` for every `id` in `data/test.csv` (80 timesteps × 7545 breaths).
Official metric: **MAE on inspiratory rows only** (`u_out == 0`); lower is better.

This folder contains the consensual swarm solution: a weighted ensemble of three
independently trained BiLSTM families with different feature sets. All models are
trained **only** on `data/train.csv`; validation/early-stopping use a holdout cut
from the training set (`fold = breath_id % 5`). No external pretrained models,
no competition-specific artifacts, no test labels were used.

## Final submission
`submission.csv` — columns `id,pressure`, 603,600 rows, row-aligned to `data/test.csv`
(== `data/sample_submission.csv` id order).

## Validation (out-of-fold, inspiratory-only MAE)
| ensemble member | family | 5-fold OOF MAE |
|---|---|---|
| `b50` — 50-physics-feature BiLSTM | 4×BiLSTM-512 → 2×BiLSTM-256, aux expiratory loss | 0.18634 (average of per-fold models + all-data seeds) |
| `c34` — 34-feature BiLSTM (canopus_omega) | 4×BiLSTM-512 → 2×BiLSTM-256 | 0.19071 |
| `ald` — 43-physics-feature BiLSTM (aldebaran_eta) | 4×BiLSTM-512 → 2×BiLSTM-256 | 0.17841 |
| **weighted blend** | | **0.17179** full OOF (leave-one-fold-out CV of weights: 0.17184) |

Independently reproduced by `canopus_omega` with `blend2.py` (same weights, OOF 0.1717865);
LightGBM stacking and per-`(R,C)`/per-step bias corrections were tested and gave no gain.

Blend weights (simplex MAE minimization on all 5 folds of out-of-fold predictions,
validated by leave-one-fold-out CV): **b50 = 0.2687, c34 = 0.1973, ald = 0.5340**.
Details in `blend_report_final.json` (created by `blend_final.py`).

Test-side variance reduction (weights unchanged, fit on the 5-fold OOF):
- `b50_test = 0.75 × combined50_test + 0.25 × mean(all-data 50-feat models, seeds 777/999/555)`
- `ald_test = 0.75 × mean(5-fold test) + 0.25 × all-data t3 model`
- `c34_test = combined34_test`

## Solution code
- `preprocess2.py` — feature engineering for the 50-feature `b50` family (vectorized
  over `(n_breaths, 80)` arrays). Writes `X2_train.npy`, `X2_test.npy`,
  `feature_meta2.json`.
- `train_lstm3.py` — model/training for the 50-feature family: 4×BiLSTM-512 →
  2×BiLSTM-256 → MLP head, target standardization, masked L1 with expiratory
  auxiliary weight 0.1, OneCycle, AMP fp16, `fold = breath_id % 5`.
- `blend_final.py` — loads row-aligned OOF/test predictions of the three members,
  fits simplex MAE weights on out-of-fold data (with leave-one-fold-out CV),
  writes `submission.csv`.
- `run_all.sh` — end-to-end reproduction sketch.

The other two members (`c34`, `ald`) were produced by the peer agents using their
own feature caches; their scripts/artifacts live in
`workspaces/canopus_omega/` and `workspaces/aldebaran_eta/`. Their test/OOF
predictions are consumed by `blend_final.py` (row-aligned to train/test order).

## Key modelling findings (shared on the blackboard)
1. **Target standardization** before L1 loss gives a clear optimization win.
2. **Auxiliary expiratory loss** (`u_out==1` weighted 0.1) helps generalize vs a
   hard inspiratory-only mask.
3. **C-normalized cumulative flow** features (`cumsum(u_in)/C`, `cummean/C`,
   `u_in/C`, `cumsum/(R*C)`) and **`u_out_lead`** (inspiration-end transition)
   are important physics features.
4. High-capacity LSTMs **memorize individual breath waveforms** (train MAE ≈0.09
   vs val ≈0.19); the cure is not more regularization (it underfits) but
   feature quality, auxiliary losses, architecture, and **ensembling**.
5. Ensemble of diverse feature families beats any single model (0.186 → 0.172).

## Environment
Python 3.11, PyTorch 2.2.0+cu121, NumPy, pandas, scipy. No external checkpoints
were downloaded or used.

## Note
During development one helper script printed expiratory MAE as the validation
number; the final metrics above are the correct inspiratory (`u_out==0`) MAE and
were independently cross-checked from the saved prediction arrays.
