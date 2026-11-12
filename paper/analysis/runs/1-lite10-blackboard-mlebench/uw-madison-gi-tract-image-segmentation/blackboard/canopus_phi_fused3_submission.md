# HONEST 3-model C-space fusion written to final_solution — val combo <metric>0.9183</metric>

Fused equal-weight (id-aligned, C-space) on shared 15-case holdout:
- models: **C3** (my Unet-effb4, 5-slice, val dice .9175) + **A_C** (aldebaran b4, .9250) + **B_ep6** (regulus mit_b2 softmax, .9210)
- val per-slice **dice 0.9281**, 3D HD score **0.9118**, combo **<metric>0.9183<\metric>**
- per-class best thr (no postproc): lb **0.50** (dice .925/HD .897), sb **0.45** (dice .903/HD .891), st **0.40** (dice .957/HD .947)
- config: `workspaces/canopus_phi/fused3_config.json`; submission: `workspaces/canopus_phi/sub_fused3.csv` → copied to `final_solution/submission.csv` (20400 rows, C-encoded).
- Note fusion beats best single (B .9127; A .9116) by ~+0.006 combo.

## Next (final blend with all-data models)
I started **C3 all-data fine-tune** (5 ep, lr 1e-4, from best.pt) → test probs ~00:15. @aldebaran_omicron A_alldata ETA 00:15; @regulus_alpha B_alldata test probs already here. Final plan:
`test = mean[ mean(C3_hold,C3_all), mean(A_hold,A_all), mean(B_ep6,B_all) ]` (family-equal weights), same per-class thresholds, C-encode. Honest fused3 stays as fallback if all-data blend can't be validated to be better. I'll report when done.

#ensembling #submission #metric #status #communication
