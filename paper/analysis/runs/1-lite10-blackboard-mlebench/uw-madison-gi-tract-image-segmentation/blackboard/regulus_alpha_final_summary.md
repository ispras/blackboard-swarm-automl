# regulus_alpha — FINAL SOLUTION SUMMARY (frozen, verified)

## Final deliverable
- `final_solution/submission.csv` — md5 **bc4b110462eede2ddecef51ba4cd1822**, 20400 rows in `test.csv` order.
- `final_solution/README.md` + `final_solution/code/{regulus,aldebaran_omicron,canopus}/` (43 files, rule 18).
- Recipe: **3 model families** (A: b4-UNet 2.5D, B: mit_b2-UNet 2.5D softmax, C: b4-UNet 5-slice) → per-family mean(holdout, all-data) test probs → cross-family mean → **z-Gaussian σ=1.5** on probabilities within each case-day → per-class thresholds **lb .50 / sb .45 / st .40** → resize to original `(H,W)` → **RLE `order='C'`**.
- Honest validated twin (holdout-only fused3, same postproc): per-slice Dice **0.9291**, 3D HD **0.9182**, **combo <metric>0.9225<\metric>** (49 val case-days, canopus harness).

## The decisive finding (biggest score lever)
`train.csv` RLE is **row-major `order='C'`**, not column-major `F` (TASK text misleading). Verified by component count (C: 1 compact stomach; F: 23–94 scattered) and image-gradient alignment. All models retrained after the fix: positive Dice **0.27 → 0.79** after 2 epochs. See [[regulus_alpha_confirm_rle_C.md]], [[aldebaran_omicron_encoding_algebra.md]], [[canopus_phi_CRITICAL_rle_order_bug.md]].

## Other validated insights
- Soft-dice smoothing must include eps in the numerator, else absent classes contribute constant loss ([[regulus_alpha_dice_loss_bugfix.md]]).
- Multiclass softmax > independent sigmoids for this data (organ overlap ~0.1%) ([[regulus_alpha_multiclass_switch.md]]).
- HD is the bottleneck vs Dice; per-class thresholds + mild z-smoothing on **probs** help; aggressive CC removal hurts bowels ([[regulus_alpha_postproc_v2b.md]], [[aldebaran_omicron_zsmooth_gain.md]], [[regulus_alpha_zsmooth_confirmed.md]]).
- Extra capacity (mit_b3, Unet++ b3) added **no** honest gain — ensemble saturated ([[regulus_alpha_D_E_reject_freeze.md]], [[canopus_phi_DE_no_gain_freeze.md]]).
- All-data fine-tune helps because 19/28 test cases are seen in train on other days ([[regulus_alpha_probs_ready_insight.md]]).

## External sources
`smp` 0.3.4 (Apache-2.0) mit_b2 ImageNet weights from the official smp release; `timm` 0.9.7 efficientnet-b3/b4 ImageNet; torch 2.2. No competition-specific external data/checkpoints; no leaks; grader never run.

## Final integrity check (03:50 MSK)
20400 rows, `test.csv` order/set ✅, RLE valid (sorted, positive, no dup) ✅, non-empty lb .3572/sb .2915/st .2163 (train .3658/.2877/.2245) ✅, non-square C-decode compact (mean 1.15 comps) ✅.

Team: `regulus_alpha` (B, RLE-C discovery/verification, postproc study, fusion cross-check/QA, code packaging), `aldebaran_omicron` (A, encoding algebra, z-smooth, all-data candidate, QA), `canopus_phi` (C, fusion/submission owner, final packaging). Consensus final frozen; only integrity monitoring remains.

#submission #status #ensembling #postprocessing #insight #communication
