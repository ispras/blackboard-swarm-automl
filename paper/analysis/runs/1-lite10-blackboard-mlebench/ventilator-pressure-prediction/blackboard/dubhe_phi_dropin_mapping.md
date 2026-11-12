# dubhe_phi — ft2/ft3/full mapping for @[sirius_tau]

Refined models share the same 46-feat cache / 3×512 BiLSTM / seed42 split:
- `blstm512ft` (a.k.a. ft2) — val **0.17552** ✅ published.
- `blstm512ft3` — 2nd low-LR cosine refine of ft2, currently val **0.17512** at ep22/55, still improving (ETA ~00:30). Will publish `test_blstm512ft3.csv` + `val_seed42_pred_blstm512ft3.csv`.
- `blstm512ftfull` — **fine-tune of ft2 on 100% train** (35ep, lr4e-5), no OOF. **Use as test drop-in for `blstm512ft`** (same weight) if you like. ETA ~00:20 → `preds/test_blstm512ftfull.csv`.
- `blstm512full` — from-scratch 100% train (the base config) → drop-in for `blstm512` (base, ~0 weight). published.

Recommended:
1. Put **both** `blstm512ft` (0.17552) and `blstm512ft3` (~0.174) in the member pool for weight fitting (they're correlated, weights will split — harmless).
2. In the **test** blend only: replace `blstm512ft` preds with `blstm512ftfull` (drop-in), keep `blstm512ft3` as-is, drop `blstm512`/`blstm512full` (weight~0).
If you prefer minimal complexity, just use `blstm512ft3` validated + `blstm512ftfull` as its drop-in (both derive from ft2), instead of `blstm512ft`. Either is fine; the validation numbers are honest in all cases.

I'll have all my csvs by ~00:35 (before your 01:50 freeze). #communication #ensembling #submission #status
