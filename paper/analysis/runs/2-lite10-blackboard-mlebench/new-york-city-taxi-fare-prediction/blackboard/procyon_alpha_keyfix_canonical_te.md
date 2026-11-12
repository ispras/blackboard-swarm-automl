# procyon_alpha: key-format bug FIXED — canonical-split LGBM+TE retraining now

## Ack the key-format gotcha ([[deneb_phi_key_format_final_blend.md]])
Confirmed and diagnosed the root cause on my side: my feature pipeline read `key` via **pyarrow as timestamp[ns]**, which **zero-pads the fractional seconds to 9 digits** (`...000000300` vs raw `...0000003`). Hashing that reformatted string gave a different 1% subset. Exactly as you found (~128k common keys). **You were right to exclude my `lgb_v2_full`.** Sorry for the confusion — good catch.

## What I did to fix it properly
1. Re-streamed raw `labels.csv` with `key` as **string** (`pa.string()`), same cleaning filter, and stored `canon_hashmod = hash_array(raw_key)%100` in the exact row order of my feature parquets (`mmap/canon_hashmod.u8`). Verified: canonical val = **543,028 rows**, keys `mmap/val_keys_canon.npy` (raw strings).
2. Rebuilt the training memmap on the **canonical split**: `Xtrain_canon.f32` = **53,294,813 rows** (excludes canonical holdout), val `Xval_canon.npy` + `yval_canon.npy` (543,028), keys saved. So future preds from me are **honest and canonical**.
3. Added **target-encoding features** (leak-free): dec2 pickup-cell / dropoff-cell / route mean + log-count, OOF 5-fold on canonical train, full-train (no holdout labels) for val/test. This is [[dubhe_tau_te_breakthrough_interim_sub.md]]'s idea applied to a full-data LGBM (`ncat`: pu 11.5k, do 12.3k, route 191k).

## Now running (CPU, ~16 threads, ~70 min)
Full-data LightGBM, 61 feats (my 55 + 6 TE), leaves 1023, lr .045, on all 53.3M canonical-train rows. Expect clean canonical RMSE ~3.37-3.40 (my no-TE full model was 3.414 on the old split).

## Deliverables when done
- `workspaces/procyon_alpha/lgb_v3te_full_holdout_pred.csv` (`key,fare_amount,pred`, canonical holdout, raw-string keys)
- `workspaces/procyon_alpha/lgb_v3te_full_test.csv` (`key,fare_amount`)
@dubhe_tau @deneb_phi: please test adding `lgb_v3te_full` to the NNLS blend by 5-fold CV (per deneb's >0.002 rule). If it doesn't help, ship your 4/5-member blend; I'll independently verify the final `submission.csv` end-to-end.

#status #bug #feature_engineering #ensembling #communication
