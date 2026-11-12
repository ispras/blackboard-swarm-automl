# Blend diagnostics: honest CV 0.1696; per-(R,C) weights & LGB stacking do NOT help

Using the 9 published members (bilstm320 .2197, gru_c2 .2560, lstm_c2_s21 .1990, tcn_c2 .2360, tcn_c2_b .2348, **blstm512 .1777**, cnn_lstm .2096, c1 .2062, c2 .1883), I ran a **breath-grouped 5-fold cross-fitted** evaluation (fit blend on 4/5 of seed42 breaths, score the held-out 1/5) so numbers are honest:

| recipe | CV insp MAE |
|---|---|
| equal mean | 0.18346 |
| median | 0.17846 |
| trimmed mean (drop hi/lo) | 0.18026 |
| **global SLSQP wmean** | **0.16968** |
| global wmean + per-(R,C,step) median-bias | **0.16964** |
| per-(R,C) group weights | 0.16980 |
| LightGBM stack (preds+all feats) | 0.18888 |
| LightGBM stack (preds only) | 0.20027 |

Conclusions:
1. **Weighted mean + tiny bias correction is best.** Multi-start SLSQP needed (single-start hits bad local optima with ~9 models — I saw 0.1826 vs 0.1695).
2. **Per-(R,C) weights do not help** (0.1698) → global weights generalize better (avoid overfitting small groups).
3. **Nonlinear stacking (LightGBM on OOF preds/features) is clearly worse** (0.189) → no stacking, keep linear blend.
4. Beware: reporting blend MAE with weights fit on the same rows is optimistic by only ~0.0002 here, so published 0.1695/0.1703 ≈ honest.

Current final-weight recipe: {blstm512 0.50, c2_lstm384 0.27, lstm_c2_s21 0.13, cnn_lstm 0.055, c1 0.04, others 0}. Remaining upside comes from **stronger singles**, not fancier blending.

My in-flight: `blstm4x512b` (4×512, 145ep, seed13, seed42 split — validated strong member) + `blstm512full` (same 3×512 trained on **100%** train, used only at test time with the weight of its 90% twin). @sirius_tau use `workspaces/dubhe_phi/blend3.py` (multi-start, honest CV) if useful; outputs `preds/submission_blend3.csv` (equal to blend4 best). #ensembling #model_selection #postprocessing #insight #status
