# capella_pi — model card / final artifact note (for final_solution README)

## What capella_pi contributed
Level-2 learned stacker + post-processing + validation protocol. Own tabular LightGBM baseline.

### 1. capella tabular LightGBM (level-1, independent)
- Model: **LightGBM 4.5.0** (MIT License), CPU, no pretrained weights.
- Features (`features.py`): tracking pair geometry, per-player nearest-neighbour context from all-pairs, helmet-box pair geometry per view (Sideline/Endzone) incl. IoU/normalized distance/lower-helmet cue, +/-1/2-step lags/leads and rolling mean/max.
- 5-fold GroupKFold by `game_play` (folds from `workspaces/alkaid_psi/cv_folds.csv`).
- OOF: AUC 0.9945, MCC 0.7049 (raw global). Artifacts: `oof_lgbm.parquet`, `test_lgbm.parquet`, `train_feat.parquet`, `test_feat.parquet`.

### 2. Learned stacker (level-2, used in final blend)
- LightGBM level-2, separate PP/Ground models, 3-seed average, on top of OOF/test probs from all agents + temporal-neighbour features + model disagreement + canopus `ground_extra` (in v5).
- **No external pretrained checkpoints used by capella_pi.** Only LightGBM (MIT) + pandas/numpy/scikit-learn (BSD). All training labels come from `data/train_labels.csv`; test probs come from other agents' fold-averaged models.
- Inputs (probs): canopus `ens_v4` p_pp/p_gr, pp v1-v5, ground v1-v4, ground GRU v1-v4, pp GRU v1-v3, pp/ground XGB1; alkaid CNN v2m, ground CNN v1m/vB1m/vC1m; capella lgbm.
- Selection: GroupKFold OOF MCC (threshold + centered temporal smoothing swept on OOF), plus **nested-threshold check** (choose th/window on 4 folds, apply to held-out fold) to guard against threshold overfit.

### 3. Final mix (recommended source)
`0.4*logit(stack_v4) + 0.3*logit(stack_v5) + 0.3*logit(canopus ens_v4)` -> **OOF 0.7850, nested-threshold 0.7820**.
- Code: `stack_v3.py --rebuild --seeds 3 --tag v5`, `build_mix.py`, `make_submission.py`, `nested_threshold_eval.py`.
- Artifacts: `oof_stack_mix.parquet`, `test_stack_mix.parquet`, `submission_stack_mix.csv` (463,243 rows, sample order).

## Reproduction
All random seeds fixed; no external data; internet not used. See `stack_v3.py` header logic and `features.py`.
External model log for capella_pi: **none** (no pretrained checkpoints).
