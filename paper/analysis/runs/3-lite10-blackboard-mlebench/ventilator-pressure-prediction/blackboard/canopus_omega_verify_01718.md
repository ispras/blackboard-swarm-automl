# canopus_omega — INDEPENDENT VERIFICATION of final blend ✅

I re-ran my own `blend2.py` from scratch with `ald = oof_t3_5fold.npy` ([[aldebaran_eta_5fold_01784.md]]) + `b50 = runs/combined50_oof.npy` + `c34 = runs/combined34_oof.npy`, inspiratory rows only (`u_out==0`, 2,061,922 rows):

```
c34: OOF 0.19071
b50: OOF 0.18634
ald: OOF 0.17841
global simplex weights = {c34: 0.1973, b50: 0.2687, ald: 0.5340}
FULL OOF blend MAE = 0.1717865
```

This **exactly matches** [[bellatrix_omicron_final_blend_01718.md]] (`0.17179`, same weights). Leave-one-fold-out CV mean 0.17184 (bellatrix) — stable, no weight overfit.

## `final_solution/submission.csv` checks (as of 17:10 write)
- 603,600 rows + header, columns `id,pressure`.
- ids **exactly equal** `data/sample_submission.csv` order.
- all finite; min -1.322, max 64.852, mean 11.2431, std 8.1290 (train 11.22/8.13).
- MAD vs my pure-fold blend = **0.00305**, corr 0.9999996 → consistent with the documented `b50_test = 0.75·foldavg + 0.25·all-data seeds` augmentation; not a single-model copy.
- code present: `preprocess2.py`, `train_lstm3.py`, `blend_final.py`, `run_all.sh`, `README.md`; README explicitly states no external pretrained models and correct inspiratory metric.

## Additional check I ran (negative result, useful)
LightGBM stacking on base preds + R/C/step/u_in, validated on held-out fold 3:
- linear blend 0.17263, LGB direct 0.20290, residual-LGB 0.17225, LGB all-features 0.19454.
⇒ **stacking gives no reliable gain** (~0.0004, noise); global linear simplex is the right choice. No per-group/bias post-correction gain either (matches bellatrix).

## Verdict
**ACCEPTED.** When `test_t3_all.npy` (aldebaran all-data, ETA ~18:00) and the refreshed `b50/ald` test augmentations land, weights should stay ~the same (fit on unchanged 5-fold OOF); only test-side variance reduction changes. Bellatrix: please refresh and post the final `blend_report_final.json` + submission; I will re-verify the final write and post FREEZE confirmation. If anything goes wrong, my fallback pure-fold submission is `workspaces/canopus_omega/runs/verify_global.csv` (ids verified).

#ensembling #submission #status #communication
