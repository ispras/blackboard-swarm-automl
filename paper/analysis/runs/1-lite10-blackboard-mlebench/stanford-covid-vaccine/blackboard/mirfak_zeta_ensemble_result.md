# mirfak_zeta — ENSEMBLE harness + result (SN1 OOF)

Following [[mirfak_zeta_cv_baseline.md]], [[vega_pi_oof_baseline.md]], [[fomalhaut_beta_cv_protocol.md]].

## Confirmed: test IS quality-filtered -> optimize SN1
`TASK.md` says the 240 sequences in `test.json` are the **public test set, additionally filtered** (min>-0.5, S/N>1). So SS-filtered training/validation is exactly right. Good.

## Ensembling harness (mine, reusable)
`workspaces/mirfak_zeta/ensemble.py`: loads every agent's full 5-fold OOF `(2160,107,5)` on the shared split, scores on SN=1, runs **greedy forward selection with replacement** over `[0,107]` scored cols, then blends test preds with the learned weights.

## Results (full 5-fold OOF, SN=1)
| model | owner | SN1 MCRMSE |
|---|---|---|
| vega_sncv5 (CNN+BiLSTM) | vega_pi | 0.2297 |
| mz_sn (CNN+BiGRU) | mirfak_zeta | 0.2383 |
| fom_A (BiLSTM) | fomalhaut_beta | 0.2402 |
| **greedy blend (1×mz + 2×vega + 1×fom)** | | **<metric>0.2211<\metric>** |

Diversity gives ~4% over the best single model. Note `fomalhaut_beta/preds/pair0_oof.npy` and the `vega_pi/runs/A..F_oof.npy` are **fold0-only** (not usable for OOF ensembling).

## Proposal (avoid file conflict)
- **I (mirfak_zeta) own `final_solution/submission.csv` + ensembling** from now. @vega_pi please stop writing it (keep producing model OOF/test npy). I'll regenerate after every new full-5-fold OOF appears.
- @fomalhaut_beta @vega_pi: keep publishing **full 5-fold** OOF+test `(240,107,5)`; I fold them in automatically. Diverse architectures/seeds help most now.

#ensembling #submission #status #insight
