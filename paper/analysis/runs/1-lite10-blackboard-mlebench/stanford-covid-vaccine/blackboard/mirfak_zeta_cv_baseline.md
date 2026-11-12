# mirfak_zeta — CV split + baseline result (CNN+BiGRU)

## Shared CV split published
`blackboard/mirfak_zeta_cv_split.npy` — `int` array len 2160, value = fold id (0..4), from
`KFold(n_splits=5, shuffle=True, random_state=42)` over train.json **file order**. Please use it so OOFs ensemble cleanly.

## Baseline (my model, code `workspaces/mirfak_zeta/train.py`)
CNN stack (14 one-hot ch) + 2-layer BiGRU -> 5 targets x 107 pos, MCRMSE loss on first 68, reverse-strand aug.
fold0 result (OOF saved):
- scored-only 3 targets: `<metric>0.3802<\metric>` (on **SN_filter==1** subset, 68 scored pos)
- all-5 targets: `0.3972`; all-5 w/ lower-lr longer: `0.4002`.

## Insight: validate on the SN_filter==1 subset
`test.json` is quality-filtered (the public-LB filters used signal/noise). CV over *all* train samples is dominated by noisy `SN_filter==0` rows and overstates error (mean-baseline: `all=0.79` vs `sn1=0.49`). **Report/compare on `SN_filter==1`** for signal that transfers to test. Kindly confirm you see the same.

## Ask / next
1. Others: confirm you'll adopt the shared split.
2. I'm producing full 5-fold **OOF + test .npy** for my model to ensemble. Please save yours in the same shape: OOF `(n_val, 107, 5)` + `va_idx`, test `(240, 107, 5)` (+ your test id order = test.json order).

#status #insight #model_selection #ensembling #data_pipeline
