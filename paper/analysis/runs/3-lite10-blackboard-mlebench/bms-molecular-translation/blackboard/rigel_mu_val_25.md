# rigel_mu — val <metric>25.90<\metric> @step12000 (rigel alone now ≈ hybrid level); training solo, 231 img/s

Refs: [[alderamin_xi_rigel_dominates.md]] [[rigel_mu_extend_training.md]]

## Trajectory (1024 holdout, raw greedy, no validity gate)
| step | 7k | 8k | 9k | 10k | 11k | **12k** |
|---|---|---|---|---|---|---|
| Lev | 40.6 | 35.9 | 31.5 | 33.7 | 34.0 | **<metric>25.90</metric>** |

The 10k/11k bump was LR-restart noise; it is dropping fast again. Loss 0.635, LR now annealing (1.87e-4 -> 0).
- GPU is now **solo** (TTA procs finished/exited): 33.5 GB used, 47 GB free, **231 img/s**, ETA schedule end ~**16:50**.

## Implications for the final rule
- rigel-alone 25.9 is already world-class for this task; the hybrid **`1m-valid ? 1m : rigel`** should still be the best (1m is even better on the ~80% valid rows), but with a **much stronger rigel on the invalid rows** the hybrid should now be **well below 24.9**.
- I'll still deliver `rigel_val20k.csv` + `rigel_test_invalid97k.csv` at ~16:45, and if the trend holds I'll also produce **`rigel_test_all.csv`** (all 484,838) so you can choose per-row between rigel and 1m freely — at ~231 img/s training-time GPU but inference is decode-bound (~100+ img/s => ~1.3 h). I'll only do all-test if budget allows after invalid97k.

Tags: #status #training #ensembling #inference
