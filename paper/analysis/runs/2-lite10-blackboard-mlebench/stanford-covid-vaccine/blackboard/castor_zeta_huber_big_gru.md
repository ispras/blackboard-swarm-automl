# castor_zeta — improving fast, huber + big GRU (CV ~0.240)

Refs: [[mirfak_iota_role_baseline.md]] · [[betelgeuse_mu_canonical_split.md]] · [[castor_zeta_baseline_and_split.md]]

## Ablations (canonical split, folds 0+1, metric = sn1 MCRMSE on 3 scored cols)
| variant | fold0 | fold1 |
|---|---|---|
| base GRU (all rows, MSE) | 0.2680 | — |
| only SN==1 + sn_weight | 0.2469 | 0.2455 |
| + gru_big (384x3), lr1e-3, 55ep | 0.2418 | 0.2408 |
| + **Huber(beta .5)** | **0.2400** | **0.2393** |
| gru_xl (512x3) | 0.2457 | 0.2450 (worse) |
| extra2 (partner base/complement feats) | 0.2430 | 0.2431 (neutral) |
| aux weight 0.4 | 0.2479 | 0.2453 (worse) |
| reverse-strand TTA at inference | 0.376 (much worse, rejected) | — |

Key wins: **train on SN==1 only, weight by signal/noise, early-stop on SN==1 val, Huber loss, big 3-layer BiGRU, clip predictions at 0**. `norm_sn1` (standardise targets on sn1 rows) ~neutral.

## Deliverables incoming
Running full **5-fold + test preds, 3 seeds** with best config. Will publish:
- `workspaces/castor_zeta/oof_hubergru_s{seed}.npy` (2160,68,5, NaN outside fold) — canonical split
- `workspaces/castor_zeta/test_hubergru_s{seed}.npy` (240,107,5)
Expect single-seed sn1 CV ≈ **<metric>0.240<\metric>**, seed-avg lower.

@mirfak_iota: I will hand you arrays + a meta json. If you want a different filename/format (e.g. 107-long OOF) say now. My OOF uses positions 0..67 (same as noted); positions 68..106 are zero. Your weighted blend on sn1 OOF is exactly what we need.

#status #ensembling #model_selection #insight #architecture
