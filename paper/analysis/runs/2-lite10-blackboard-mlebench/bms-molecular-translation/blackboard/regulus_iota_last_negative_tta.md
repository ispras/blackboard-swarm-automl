# regulus_iota — last-hour negative result: small-angle TTA does NOT help (rot180 stays the only useful iota TTA); final unchanged

Refs: [[regulus_iota_final_signoff.md]] [[regulus_iota_final_robustness_check.md]]

Tried ±10° rotations of the phase-3 model as extra medoid voters on the first-3000 holdout:
- standalone first-3000 Lev: p3 = 1.773, rot180 = 5.232, **+10° = 12.593, -10° = 11.981**.
- medoid (q8, replacing rot180): +10° -> 1.4103; -10° -> 1.3560; +10/-10 both -> 1.3507.
- adding +10° alongside rot180 -> 1.2920 (worse than the frozen 1.2507).
=> Small-angle TTA is out-of-distribution for this model; **no change**. `final_solution/submission.csv` remains md5 **b5b6a24cb5c29b4830a3f8e9a69a005f**, holdout **<metric>1.2507<\metric>**.

#insight #augmentation #ensembling #model_selection #status
