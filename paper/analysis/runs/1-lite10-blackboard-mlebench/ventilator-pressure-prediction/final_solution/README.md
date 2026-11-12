# Google Brain — Ventilator Pressure Prediction

Metric: **MAE** of `pressure` during the **inspiratory phase** (`u_out==0`).
Submission: `submission.csv` — columns `id,pressure`, 603,600 rows (aligned to `data/sample_submission.csv`).

## Solution summary
A swarm-trained ensemble of bidirectional sequence models predicting all 80 timesteps of each breath.

1. **Feature engineering** (`code/fe2.py`, `code/fe3.py`, `code/fe.py`) → per-breath arrays `(n_breaths, 80, F)`:
   - base `u_in, u_out, time_step, step, dt, R, C, RC, R+C, R/C`
   - physics: volume proxy `V=cumsum(u_in*dt)`, `V/C`, `R*u_in`, `phys_pressure=R*u_in+V/C`, `u_in/C`, `u_in/(R*C)`
   - temporal: lags 1–12, leads 1–5, diffs, diff·dt, rolling mean/std/max/min (win 3–9), cumulative mean/max/min, cum_frac, u_in_frac, total volume
   - R/C one-hots (R∈{5,20,50}, C∈{10,20,50})
2. **Models** (`code/train.py`): 3–4 layer BiLSTM, hidden 384–512, dropout 0.1, residual input skip. Weighted L1 loss (`u_out==0` weight 1.0, `u_out==1` weight 0.1). AdamW + cosine / ReduceLROnPlateau. Early stopping on holdout inspiratory MAE; strong models get extra low-LR cosine "refine" passes.
3. **Ensembling** (`code/final_blend.py` + `code/manifest_final6.json`): non-negative simplex weights via multi-start SLSQP minimizing inspiratory holdout MAE. Inspiratory rows = weighted blend; expiratory rows = separate blend of exp-capable members (insurance). Final predictions clipped to the training pressure range.

## Validation
Holdout = random 10% of **breaths** (seed 42, 6,790 breaths), scored only on `u_out==0` rows. All training/early stopping/weight fitting used only this cut of `data/train.csv`; **test labels were never accessed**. Final blended holdout inspiratory MAE = **0.16037** (honest cross-fitted = 0.16089).

## Files
```
submission.csv                 final predictions
code/fe.py, fe2.py, fe3.py     feature pipelines (caches 28/46/59 features)
code/train.py                  model definitions + training loop
code/finetune_full.py          full-data (100% train) low-LR fine-tune for test drop-in
code/infer.py                  re-inference from a checkpoint
code/export_run.py             export holdout/test predictions in blend format
code/final_blend.py            ensemble fitting + submission writer
code/manifest_final6.json      final member list, weights source, test-pred paths
code/README_models.md          per-member architecture, holdout MAE, weight, sources
```

## External resources
**None.** No pretrained checkpoints, no external data, no solution write-ups. All models trained from scratch on `data/train.csv`; only standard libraries (PyTorch, NumPy, pandas, SciPy, scikit-learn) were used. The ensemble includes models from peer agents in the same swarm, all trained on the same provided training data (see `code/README_models.md`).
