# regulus_iota — last-hour TTA sweep finished: angle AND scale TTA are both negative; final untouched

Refs: [[regulus_iota_last_negative_tta.md]] [[regulus_iota_final_signoff.md]]

Standalone holdout Lev on first-3000 (phase-3 model, greedy):
- p3 = **1.773**; rot180 = 5.232; ±10° = 12.59 / 11.98; **scale 0.85x = 32.57; 1.15x = 11.61**.
Medoid variants using angle/scale as extra voters were all >= frozen 1.2507 (1.29-1.41).
Conclusion: only rot180 (already in the frozen ensemble) is useful as an iota TTA voter. No change to the submission.

`final_solution/submission.csv` md5 **b5b6a24cb5c29b4830a3f8e9a69a005f** (holdout <metric>1.2507<\metric>) — frozen and verified. No jobs running; hard-end ready.

#insight #augmentation #ensembling #status
