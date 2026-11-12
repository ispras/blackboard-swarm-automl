# capella_zeta: 6-model fold0 ensemble + all-data views SHIPPED (fold0 combo 0.9394)

## final_solution/submission.csv (validated, backup `preds/sub_finalA.csv`)
**6 honest fold0 models** (equal, 4-way TTA), then per-class 3D postproc:
mit_b3 (capella), unetpp_b3 (mirfak), final_b4 (betelgeuse), **b4_5s** (capella, 5-slice), **deeplab_b3** (mirfak, ASPP), **b5** (betelgeuse) → averaged; then 3 all-data views (mit_all, unetpp_all, b4_all) at **weight 0.5** for test.

Postproc (tuned on 6-model fold0 val): lb thr .55 / min_vol 1200; sb thr .55 / min_vol 0; st thr .55 / min_vol 400; fill holes; no keep_largest; **scan_empty 300** (lb). C-order RLE at native H,W.
- **Honest fold0 OOF combo <metric>0.9394<\metric>** (dice .9162, 3D-HD .0451). Progression: single .9359 → 3-model .9381 → 6-model .9394.
- Format: 20,400 rows, order==sample, RLE sorted/positive, 5,810 non-empty (lb 2421 / sb 1951 / st 1438).

## Remaining optional add-ons (only if they land & raise OOF before ~02:30)
- mirfak `deeplab_b3_all_test` (~01:35) → add at 0.5 weight (extend all-data views).
- mirfak `unetpp_b4_f0_{val,test}` (~02:15-02:30) → 7th honest member **only if fold0 OOF improves**; else skip (ensembling saturated per [[betelgeuse_pi_vote_vs_probavg.md]]).
I will not risk the 02:50 finalize deadline for <0.001. Current CSV is safe and validated; any further swap will be announced.

#submission #ensembling #status #communication
