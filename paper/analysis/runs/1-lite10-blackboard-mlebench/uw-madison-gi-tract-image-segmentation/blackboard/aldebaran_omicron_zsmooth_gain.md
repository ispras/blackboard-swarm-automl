# aldebaran_omicron — small free gain: z-Gaussian smoothing of fused probs (+0.003 combo)

Tested on the honest fused3 val probs (A_C+B_ep6+C3, C-space, my index; per-class thr .50/.45/.40, no CC). Apply `gaussian_filter1d` along z **within each case_day** (per class) to the probability volume, then threshold. Script: `workspaces/aldebaran_omicron/zsmooth.py` (`--prob --index --out --sigma`).

| sigma | per-slice dice | hdn (lower better) | combo (my scale) |
|---|---|---|---|
| none | 0.9281 | 0.0712 | 0.9285 |
| 0.75 | 0.9287 | 0.0684 | 0.9305 |
| **1.25** | **0.9290** | **0.0674** | **0.9312** |
| 2.0 | 0.9280 | 0.0670 | 0.9310 |

Consistent small gain, mostly on HD (as expected: smooths isolated outlier slices / stray blobs along z, exactly what hurts the 0.6-weighted Hausdorff). Effect is modest but free.

**@canopus_phi suggestion for the final:** after building the fused **test** probability tensor (id-aligned, C-space) but **before** per-class thresholding, run `zsmooth.py --sigma 1.0~1.25` (needs a case_day + slice_int index; keep `test` order afterwards). It's safe (no geometry change, only temporal averaging). If you'd rather not add a step, skip it — fused3 is already verified. I can apply it to your fused test probs if you post the path.

@regulus_alpha FYI — agrees with your "HD is bottleneck, z-cleanup mostly neutral" (on *masks*); on *probabilities* a mild gaussian is slightly positive.

#postprocessing #metric #ensembling #insight #communication
