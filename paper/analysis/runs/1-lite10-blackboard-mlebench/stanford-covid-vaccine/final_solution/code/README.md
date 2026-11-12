# OpenVaccine mRNA degradation — final solution (swarm `mirfak_zeta`, ensembling lead)

Task: predict per-base degradation at the first 68 positions of 107-nt RNA sequences.
Metric: **MCRMSE** over `reactivity`, `deg_Mg_pH10`, `deg_Mg_50C` (per `TASK.md`).
Data: `data/train.json` (2160, JSONL), `data/test.json` (240, JSONL).

`test.json` is the quality-filtered public test split, so all development used the
**`SN_filter==1`** subset for training and validation (see `blackboard/fomalhaut_beta_cv_protocol.md`).

## Final result
Ensemble of **~100 full-*k*-fold base models** from all three swarm agents.
Blend (`nnls_calib` / `nnls_bag`, non-negative least squares per scored target + per-target affine):
**OOF MCRMSE = 0.2090** on the `SN_filter==1` subset (honest blender-CV ≈ same value → no blend overfit).

## Pipeline
1. **Features** — per-position one-hot of `sequence` (ACGU, 4) + `structure` (`.()`, 3) +
   `predicted_loop_type` (SMIBHEX, 7) = 14 channels over length 107.
2. **Base models** — CNN+BiLSTM/GRU, CNN-only, wide BiLSTM, multi-scale CNN, Transformer, GNN
   (`train.py` has the CNN+BiGRU member). Trained on `SN_filter==1` rows only, MSE masked to the
   first 68 positions, reverse-strand augmentation, k-fold CV (5/10/20/30-fold; more folds =>
   more train data per member => better, see `blackboard/vega_pi_kfold10_finding.md`).
3. **CV** — classification split `KFold(k, shuffle=True, random_state=42)` over train file order
   (`blackboard/mirfak_zeta_cv_split.npy` for k=5, `_cv_split10.npy` for k=10). Early stopping on
   the `SN_filter==1` validation subset.
4. **Full runs** (`run_cv.py`) produce OOF `(2160,107,5)` and test `(240,107,5)` per member.
5. **Ensemble** (`ensemble.py`) — **auto-discovers** every agent's full-*k*-fold `{oof,test}` pair,
   scores on SN=1, and fits NNLS per scored target (`nnls_calib` = NNLS + per-target affine
   calibration) chosen by an honest 5-fold CV of the blender. Greedy Caruana / plain mean /
   LightGBM stacking were all equal-or-worse.
6. **Submit** (`make_submission.py`) — apply the blend to test preds, write `submission.csv` in
   `sample_submission.csv` order; unscored `deg_pH10`/`deg_50C` filled with the scored analogues.

## Reproduce
```bash
pyenv shell sis-black
# one diverse member (10-fold, asinh target transform)
python workspaces/mirfak_zeta/run_cv.py --outdir <dir> --seeds 5 --epochs 120 \
    --scored_only --sn_only --select sn1 --nfold 10 --asinh_s 0.5
# blend every discovered full-k-fold OOF/test pair and write the submission
python workspaces/mirfak_zeta/ensemble.py
python workspaces/mirfak_zeta/make_submission.py
```

The final `submission.csv` was assembled with these exact scripts from all swarm members'
`{oof,test}` arrays (auto-discovered under `workspaces/*/{runs,preds}/`).

## What did NOT work (independently confirmed by the swarm)
Nonlinear/GBM meta-stacking, per-position calibration, prediction clipping, reverse-strand
test-time augmentation, input mixup, adding `SN_filter==0` rows, pseudo-labelling the test set
(fold-consistent), structure-aware/pair-distance inputs, k-NN memory model, GNN/Transformer as
diverse members. See board messages `mirfak_zeta_saturation_probes.md`,
`vega_pi_blend_and_insights.md`, `fomalhaut_beta_negative_results.md`.
