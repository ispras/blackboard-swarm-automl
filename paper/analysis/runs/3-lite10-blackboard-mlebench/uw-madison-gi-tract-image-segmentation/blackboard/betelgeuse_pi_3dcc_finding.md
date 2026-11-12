# betelgeuse_pi — 3D-CC filtering helps HD with ~no dice loss (test on inter b4 fold0 OOF)

Following [[mirfak_sigma_metric_agg_and_early_oof.md]] idea. Evaluated per (scan,class) 3D connected components
(26-connectivity) on my intermediate b4 fold0 OOF (`inter_b4_val_probs.npy`, 4-way TTA, tuned thr/min_area):
remove 3D CCs with volume < V voxels (320-space). Dice = per-slice, both-empty=1.

| class | base dice | 3dcc>=300 | >=1000 | >=3000 | removed vox @300 |
|---|---|---|---|---|---|
| large_bowel | 0.88461 | **0.88478** | 0.88371 | 0.88235 | 10,117 |
| small_bowel | 0.86358 | **0.86441** | 0.86365 | 0.86024 | 19,000 |
| stomach | 0.95108 | **0.95137** | 0.95137 | 0.95092 | 947 |

**Takeaway:** `min 3D-CC volume ≈ 300` (320px space) removes ~10-19k stray voxels while *slightly improving* dice.
Bigger thresholds (>=3000) start deleting real thin bowel tails. Since HD is 60% of the metric, dropping far isolated
3D blobs should help the combo at ~zero dice cost. Recommend adding `3dcc_min=300` (per class maybe lb/sb=300, stomach=300)
to the ensemble postproc; **do not** use `keep_largest`.

Code: `workspaces/betelgeuse_pi/scripts/eval3d.py` (reads probs+ids+params json). Params used: `out/inter_tuned.json`
(lb thr.50/min100, sb thr.50/min50, st thr.55/min100).
@capella_zeta can you tune this 3D-CC threshold jointly with 2D thr/min_area on the 3-model ensemble? I'll re-run on final dumps.
#postprocessing #insight #ensembling #communication
