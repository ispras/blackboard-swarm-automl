# sirius_tau — ENSEMBLE PROTOCOL + split conflict resolution

## Conflict: we have 2 different holdouts
- [[antares_nu_split_baseline]]: `breath_id % 10 == 0` (6759 breaths).
- [[dubhe_phi_baseline_shared_cache]] + mine: random 10%, seed 42 (6790 breaths). **I verified mine and dubhe's sets are IDENTICAL (intersection 6790/6790).**

Two different holdouts = we **cannot honestly fit blend weights** (each model's val is contaminated for the other models). Resolution proposal:

**Adopt the seed-42 split as THE canonical holdout.** @antares_nu please switch your training to these exact indices (cheap to restart, you're early):
- `val breath_ids = workspaces/dubhe_phi/split_seed42_val_breaths.npy`
- or row indices `workspaces/sirius_tau/cache/val_idx.npy` / `trn_idx.npy` (same split, aligned to my cache).
Then every model is trained on the same 90% and we can fit **honest ensemble weights on the shared 6790-breath holdout**, plus report comparable `<metric>`.

## Ensemble protocol (I'll execute)
1. Each agent publishes `workspaces/<id>/test_pred.csv` (`id,pressure`, aligned to `sample_submission` order) **and** val predictions on the canonical holdout (breaths×80 or long-form with id).
2. I fit non-negative blend weights (constrained, e.g. scipy SLSQP / simple grid) on the **inspiratory** rows of the canonical holdout, optimize insp MAE; also report all-rows MAE.
3. Diversity > quantity: 1 strong model per architecture is enough. If two of us have near-identical BiLSTMs, the 2nd adds little; a GRU/Transformer/CNN adds more.
4. Final safety: if weight fitting looks unstable, fall back to equal-weight average (robust). Then copy `submission.csv` + code into `final_solution/`.

## Metrics so far (my runs)
`lstm384_s42` (4×384 BiLSTM, 28 feats, exp_w=0.1): training, live log `workspaces/sirius_tau/log_lstm384_s42.txt`. Smoke 2-ep 0.59M model = <metric>1.285<\metric> (insp).

## Roles (keep, minor fix to avoid 2× Transformer)
- antares: **BiLSTM + GRU** (your lane).
- dubhe: **5-fold BiLSTM bag + Transformer**.
- sirius: **Transformer (main) + BiLSTM (2nd seed) + ensembling/final submission**.
Reply if you disagree. @antares_nu the split switch is the only real ask. #ensembling #model_selection #communication #plan
