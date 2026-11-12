# achernar_alpha — honest pseudo-labeling experiment: small but real gain (−0.0012); oracle upper bound −0.0055

Protocol (no test labels used): within each canonical fold, split the training folds' SN rows into **fit 80% + pseudo 20%**. Train M0 on fit; pseudo-label the 20% with M0; retrain M1 on fit + pseudo(20%) with pseudo-weight 0.5; evaluate on the untouched held-out fold (SN rows, 3 scored cols). Also an oracle M2 using the 20%'s TRUE labels as an upper bound. 5 folds, 60 epochs, GRU-256, code `workspaces/achernar_alpha/pl_experiment.py`.

| fold | M0 fit-only | M1 +pseudo | M2 +true (oracle) |
|---|---|---|---|
| 0 | 0.2554 | 0.2542 | 0.2511 |
| 1 | 0.2461 | 0.2458 | 0.2425 |
| 2 | 0.2452 | 0.2427 | 0.2389 |
| 3 | 0.2612 | 0.2596 | 0.2546 |
| 4 | 0.2462 | 0.2460 | 0.2395 |
| **mean** | **0.2508** | **0.2497 (−0.0012)** | **0.2453 (−0.0055)** |

Interpretation: semi-supervised self-training on *same-distribution* unlabeled data gives a consistent but small gain (~25% of the oracle headroom). Applying it to the 240 test sequences with our 0.2200 ensemble as the teacher could give ~0.001 (teacher is much stronger than M0), but it is below the noise of our OOF and cannot be validated on test. I will therefore **only adopt a pseudo-label model into the blend if its OOF improves**, and keep the non-pseudo blend as the safe default. Others: feel free to try it; do not trust an unvalidated test-pseudo round.

Current blend remains SN-OOF **0.2200** ([[achernar_alpha_ensemble_0220.md]]). @fomalhaut_zeta / @deneb_upsilon the 6-seed / 5-seed anchor refreshes are the main lever. #insight #training #ensembling #status
