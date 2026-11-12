# mirfak_iota — ensemble v3: `<metric>0.21625<\metric>` (my GRU scored-only+SN-weight adds a lot)

Refs: [[mirfak_iota_ensemble_v2.md]] · [[mirfak_iota_postproc_and_snweight.md]] · [[betelgeuse_mu_cv_and_leakage_finding.md]]

## New base model
`my_g512snw` = CNN + 2-layer BiGRU-512, **loss on 3 scored cols only, sample weight = signal_to_noise, all rows, 150ep, select on sn1**.
- sn1 CV **`<metric>0.22883<\metric>`** cols [0.2121, 0.2568, 0.2176] (5-fold, seed 42). Beats every single be model (best 0.22937).
- NNLS gives it the **largest weight** (0.38 / 0.40 / 0.41 per column) → the scored-only+SN-weight recipe is genuinely complementary to all-5 MSE.

## Blend
| pool | NNLS CV | + kNN gate |
|---|---|---|
| 6 be models + my_v1 | 0.22273 | 0.21987 |
| 6 be models + my_g512snw | 0.21801 | **`<metric>0.21625<\metric>`** |

Gated cols [0.1994, 0.2440, 0.2053]. **`final_solution/submission.csv` = ensemble v3** (`workspaces/mirfak_iota/ens3_test.npy`).

## Coming
- seed-2 of the same GRU, GRU-256, LSTM-512 all scoring ~0.229 individually → should push blend toward ~0.213–0.215.
- castor's hubergru (SN1-only+Huber) should add independent error.
Rerun is one command (`ensemble.py --knn-gate --clip 0`); array paths are in [[mirfak_iota_blend_plan.md]].

#ensembling #status #submission
