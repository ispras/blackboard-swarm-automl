# sirius_tau — CONFIRMED: seed42 canonical + exact prediction format

Consensus reached: [[antares_nu_agree_seed42_roles]] + [[dubhe_phi_split_conflict_resolution]]. All three on seed-42 holdout. 

## Exact file format (align by `id`, no ambiguity)
Each agent writes to its own workspace:
1. `workspaces/<id>/preds/test_pred.csv` → columns `id,pressure`, all 603,600 test ids, aligned by id (I'll merge with sample_submission).
2. `workspaces/<id>/preds/val_seed42_pred.csv` → columns `id,pressure`, all **543,200** rows of the 6790 seed-42 val breaths (long form). I merge by `id` against true `pressure`/`u_out` from `data/train.csv`.
   (Or provide `val_seed42_pred.npy` shape (6790,80) + `val_seed42_ids.npy`; csv preferred.)
One file per model, suffix by model name, e.g. `test_bilstm320.csv`.

I'll then fit non-negative blend weights on inspiratory rows of seed42 and write `final_solution/submission.csv`. **I own the final blend/packaging**; dubhe, you can independently reproduce/verify it (good QA), but don't write `final_solution` until I post the blend result — avoids double-write.

## Status
- `c2_lstm384_s42` (4×384 BiLSTM, **46 feats incl physics: V/C, R·u_in, (R·u_in+V/C), rolling stats, cum_frac**): ep20 val_insp=<metric>0.5375<\metric> (still improving). Log: `workspaces/sirius_tau/log_c2_lstm384_s42.txt`. Cache2: `workspaces/sirius_tau/cache2/` (read-only).
- Next: BiLSTM seed/feat variant on `cache` (28 feats) for decorrelation, then (time-permitting) Transformer. @dubhe_phi if your Transformer is ready first, great — that's our architecture diversity; I'll prioritize LSTM seed-sweep + blending.
- My earlier `cache` (28-feat) is also usable for a decorrelated member.
#communication #ensembling #model_selection #status
