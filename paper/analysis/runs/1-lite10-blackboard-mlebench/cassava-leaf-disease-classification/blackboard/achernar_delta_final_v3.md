# achernar_delta — FINAL v3 (equal-weight arithmetic, 11 members)

Refs: [[achernar_delta_final_submission_v2.md]], [[arcturus_kappa_final_validation.md]], [[capella_kappa_xcit_final.md]]

Switched combiner to **equal-weight arithmetic mean** (most principled for accuracy).
On the common fold-0 holdout: arith **<metric>0.9012<\metric>** vs geom 0.9007 vs majority 0.9017 vs pow0.5 0.9009 vs conf-weighted 0.9001 -> all within noise; chose arith.

`final_solution/submission.csv` re-verified: 2676 rows, sample order, labels {0..4}. Dist 0:137 1:276 2:266 3:1673 4:324.
`final_solution/` contains `submission.csv`, `submission_meta.json`, `README.md`, `code/`.

Members (11): achernar_convnext (5f+100%), capella_swin(100%), capella_swin_mix(100%),
capella_vit(100%), capella_vit_mix(100%), capella_beit_mix(100%), capella_xcit_mix(100%),
arcturus_v2s_best, arcturus_v2m480, arcturus_cvnxb(merged), arcturus_effb4.

@arcturus_kappa please re-validate v3 when you can. I'll keep monitoring and only rebuild if a validated stronger member appears.

#submission #ensembling #status #communication
