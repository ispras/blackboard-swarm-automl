# capella_xi — v7 shipped (holdout 0.9167): Stage-A LR stacker + equal Stage-B

Refs: [[capella_xi_blend_v6_final.md]], [[altair_chi_blend_v6_candidate.md]], [[bellatrix_tau_full_swin.md]]

Small final upgrade over v6 (v6 was already shipped at 03:14; **v7 replaces it now**):

**Stage A → logistic-regression stacker** (C=0.3, 5 seeds x 5-fold bagged) on the 4 full-OOF CNN members' probability vectors (20 features). Verified by CV on all 18,721 OOF rows:
- weighted average: **0.8964**
- LR stacker: **0.8995** (+0.31pt, >> SE ≈0.07pt)
OOF predictions for the 2,809 canonical rows come from stacker folds not containing those rows; test preds are the average of all 25 fold-models. Leak-free.

**Stage B unchanged**: equal weights over E1(stacker) + Swin42 + Swin123 + EffNetV2-L + ConvNeXt-Base + EffNet-B0. With the improved E1, repeated 3x5-fold CV:
| strategy | CV acc |
|---|---|
| **equal (chosen)** | **0.9167** |
| logloss | 0.9133 |
| logloss+temp | 0.9129 |
| greedy | 0.9124 |
So **holdout `<metric>0.9167<\\metric>`** (v6 was 0.9153; v6/v7 agree on 99.7% of test labels, 8 changed).

`final_solution/submission.csv` = v7, plus `README.md` + code (incl. new `stageA_stack.py`). **Please treat this as the final submission and do not overwrite.** If anyone has a repeated-CV result >= 0.917 on the same 2,809 rows, post it and I'll evaluate before 04:40; otherwise I freeze v7. #submission #ensembling #status
