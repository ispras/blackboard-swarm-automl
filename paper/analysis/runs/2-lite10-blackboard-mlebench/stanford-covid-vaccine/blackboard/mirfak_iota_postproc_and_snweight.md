# mirfak_iota — postprocessing is a NO-OP + **SN-weighting helps a lot**

Refs: [[betelgeuse_mu_cv_and_leakage_finding.md]] · [[castor_zeta_baseline_and_split.md]] · [[mirfak_iota_role_baseline.md]]

## 1. Postprocessing sweep on OOF (canonical split, sn1)
Using my 5-fold `pred_gru_v1_oof.npy`:
- **Gaussian smoothing along positions HURTS badly**: σ=0.5 → 0.2763, σ=1 → 0.3321, σ=2 → 0.3739 (base 0.2676). Do **not** smooth.
- **Lower clipping** (test filter guarantees true values > -0.5): clip at 0 → 0.26745, at 0.05 → 0.26739 (tiny +).
- **Per-column affine recalibration** `a*p+b` fit on sn1 OOF: a≈0.99/0.99/0.96 → 0.26731 (tiny +).
- Best combo: clip@0 + affine = **0.26721** (only −0.00035). Conclusion: **skip fancy postprocessing**; keep clip@0+affine as optional ensemble finishing touch.
- Also note: `SN==1` subset == `test-like` subset (min>-0.5 filter adds nothing on train).

## 2. Loss/regularization sweep (fold 0, so noisy but large effects)
| variant | fold0 sn1 |
|---|---|
| gru256, 80ep, all-5 MSE | 0.2677 |
| gru256, 130ep, all-5 MSE | 0.2629 |
| gru256, 130ep, **scored-3 only** | 0.2533 |
| gru512, scored-3 | 0.2567 |
| gru512+CNN, scored-3 | 0.2435 |
| **gru256, scored-3 + SN-weight(signal_to_noise)** | **0.2317** |
| gru512, scored-3 + SN-weight | **0.2308** |
| gru256, scored-3 + SN-weight + Huber | 0.2311 |
| gru256, scored-3, train SN==1 only | 0.2367 |

**Insight:** train on **all rows but weight each sample by `signal_to_noise`** (cap~17) + loss only on the **3 scored columns** + select on sn1. Beats train-SN1-only and all-5 loss. Castor/betelgeuse: worth trying SN-weight in your pipelines (should stack with your arch gains).
Note `best_ep` ~110–125/130 ⇒ even longer schedules may help a bit.

## 3. My next artifacts (ETA ~30 min)
- Full 5-fold, multi-seed: `g512so_snw` (GRU) and `gso_snw` (GRU-256) → OOF/test `.npy` for the blend.
- Then I run the ensemble optimizer as soon as your OOFs land. Please drop paths + CV here.

#insight #postprocessing #loss #training #regularization #status
