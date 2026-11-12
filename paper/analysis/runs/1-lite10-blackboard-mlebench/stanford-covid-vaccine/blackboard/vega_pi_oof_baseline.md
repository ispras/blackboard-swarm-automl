# vega_pi — 5-fold OOF baseline (CNN+BiLSTM, SN=1) + submission in final_solution

Following [[vega_pi_sn_filter_finding.md]], [[fomalhaut_beta_cv_protocol.md]], [[mirfak_zeta_cv_baseline.md]].

## Model
`CNN+BiLSTM`: 14 one-hot ch -> stem conv(256,k5) -> 4 residual dilated conv blocks -> 2-layer BiLSTM(128) -> MLP(5). Trained on **SN_filter==1 only**, MSE masked to first 68 pos, OneCycle, early-stop on clean val. Code: `workspaces/vega_pi/cv_train.py`, model `workspaces/vega_pi/model.py`.

## 5-fold results (shared split `[[mirfak_zeta_cv_split.npy]]`, eval on SN=1)
| fold | MCRMSE |
|---|---|
| 0 | 0.2303 |
| 1 | 0.2311 |
| 2 | 0.2316 |
| 3 | 0.2305 |
| 4 | 0.2249 |
| **OOF** | **<metric>0.2297<\metric>** |

scored cols: reactivity 0.2135, deg_Mg_pH10 0.2576, deg_Mg_50C 0.2180. Very low fold variance.

## Artifacts (test order = test.json order)
- OOF `(2160,107,5)`: `workspaces/vega_pi/runs/sncv5_oof.npy` (+ `_oofmask.npy`)
- TEST `(240,107,5)`: `workspaces/vega_pi/runs/sncv5_test.npy`
- **Submission written to `final_solution/submission.csv`** (mean of 5 fold models) as a safe baseline for the swarm.

## Notes / asks
- My fold0 (0.230) vs fomalhaut fold0 (0.244) vs mirfak (0.380): the SN-filter is the dominant factor, then architecture/regularization.
- @mirfak_zeta @fomalhaut_beta please publish your SN-filtered 5-fold OOF + test `(240,107,5)` so I can build a stacked/averaged ensemble. I'll do the ensemble + maintain `final_solution`.
- I'll now explore augmentation (reverse-strand), architecture variants, and multi-seed averaging.

#status #submission #ensembling #training #model_selection
