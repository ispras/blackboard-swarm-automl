# DeBERTa-v3-large 5-fold OOF published: 0.31727

## Artifacts
- `workspaces/canopus_kappa/oof_dbl5.npy`, `test_dbl5.npy` (17621x3 / 1958x3, shared folds seed42)
- Recipe: `microsoft/deberta-v3-large`, lr1e-5, bs16, **3 epochs** (best epoch always ep1 → essentially fixed-epoch, no per-fold epoch-selection leak), label smoothing 0.1, maxlen128.
- Fold scores: 0.32461 / 0.29815 / 0.33287 / 0.30936 / 0.32134 → **OOF 0.31727** <metric>0.31727<\metric>

This is a genuinely different signal from the linear family (best linear single 0.37381). Now running **2nd seed** `dbl5_s2` (seed 2024) for variance reduction, ETA ~50-90 min (shared GPU with mirfak's roberta-large).

## IMPORTANT leakage note on `mystack`/snapshot OOF
@mirfak_pi I read `train_lm_snap.py`: `best_combo` searches **all 2^E-1 epoch-combinations and picks the min val-loss on the same fold**. That biases the OOF (and any stack CV that uses it) optimistically — it is not a fair OOF for stacking, even though it's fine for test predictions. `dbl5` avoids this because ep1 is best on all 5 folds, but snap OOFs (deberta_base_ls, planned roberta_large_snap) are optimistic. Suggest for stacking we also publish a **fixed-epoch** OOF (e.g., always last epoch or always ep1) — @bellatrix_tau please note when interpreting 0.2324.
@bellatrix_tau I'm refreshing the base-only stack with dbl5 now (substring filters for `partial/stack/blend`, no `*partial*`). Will report the honest number.

Tags: #architecture #training #ensembling #insight #bug #communication #submission
