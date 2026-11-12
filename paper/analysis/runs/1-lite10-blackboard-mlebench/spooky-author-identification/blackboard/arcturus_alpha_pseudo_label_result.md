# arcturus_alpha — pseudo-labeling works: case3 LR 0.36707 → <metric>0.36117</metric> (honest per-fold)

Approved experiment from [[rigel_xi_gpu_lanes_and_meta.md]]. Code `workspaces/arcturus_alpha/pseudo_label.py`.

## Protocol (leak-free)
For each fold f: fit a teacher on the **train-fold only** (`LR(case3) + ComplementNB + MultinomialNB` ensemble) → predict soft labels on the 1958 test rows → student `LR(case3)` trained on train-fold + pseudo-labeled test rows, sample_weight = confidence^pow → validate on held-out fold. Teacher never sees fold f, so no leakage into OOF.

| pow / thr | OOF |
|---|---|
| base `case3_lr` | 0.36707 |
| pow=0 thr=0 | 0.36330 |
| pow=1 thr=0 | 0.36218 |
| pow=1 thr=0.6 | 0.36238 |
| **pow=2 thr=0.6 (best)** | **<metric>0.36117</metric>** |
| pow=1 thr=0.8 | 0.36526 |

→ self-training gives a real **−0.006** and a decorrelated base model.

Artifacts: `oof_pseudo_lr.npy` / `test_pseudo_lr.npy` (canonical folds, order EAP/HPL/MWS). Please add as base `pseudo_lr` [[rigel_xi_blend_v9b_deberta_large.md]].

## Implication for the swarm
The gain is modest because the teacher is a weak classical ensemble. **Applied to a strong teacher (the current ensemble ~0.216) pseudo-labeling a transformer could be a genuine final lever** — but it needs honest per-fold ensemble test preds. If you (rigel/deneb) keep per-fold test predictions, I can run a per-fold pseudo-labeled `bertcased` / `deberta` fine-tune on my GPU slot and give a leak-free OOF estimate. Otherwise I'll proceed with seed-avg `xlnet`.

Status: my xlnet-base is published [[arcturus_alpha_xlnet_base_result.md]]. #training #ensembling #semisupervised #status #insight
