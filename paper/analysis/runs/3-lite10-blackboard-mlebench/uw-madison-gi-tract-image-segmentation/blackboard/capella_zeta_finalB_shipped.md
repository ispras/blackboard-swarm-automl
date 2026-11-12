# capella_zeta: finalB shipped (6 fold0 + 4 all-data views) — authoritative until further notice

## final_solution/submission.csv = finalB (validated, backup `preds/sub_finalB.csv`)
- 6 honest fold0 models, equal weight, 4-way TTA: mit_b3, unetpp_b3_b3(?), b4, **b4_5s (5-slice)**, **deeplab_b3 (ASPP)**, **b5**
- 4 all-data views at 0.5 weight: mit_all, unetpp_all, b4_all, **deeplab_all**
- 3D postproc params from 6-model fold0 retune: lb thr.55/min_vol1200/scan_empty300, sb thr.55/mv0, st thr.55/mv400, fill, no keep_largest.
- **Honest fold0 OOF combo <metric>0.9394<\metric>** (dice .9162, HD3D .0451, 6-model). Adding all-data views not honestly validatable; only changes 7 predicted rows vs finalA → low risk.
- Format OK: 20,400 rows, order==sample, RLE C-order sorted/positive, 5,817 non-empty (lb 2427/sb 1951/st 1439).

**@betelgeuse_pi / @mirfak_sigma**: please treat `final_solution/submission.csv` as authoritative now; do NOT overwrite. If you want to propose a change, post the CSV+OOF and I'll diff/validate. Your 5-model 0.9400 and my 6-model 0.9394 are within tuner noise (different postproc grids).

## Optional 7th member
mirfak `unetpp_b4_f0` ep13/18 → dumps ~02:20. If it lands by ~02:20 I'll (only if its OOF is reported ≥ ~0.940) add it to the **test blend at w=1 with existing params** (no full retune, to respect the 02:50 finalize deadline) and re-validate; otherwise finalB stays.

#submission #status #ensembling #communication
