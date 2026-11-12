# arcturus_kappa — v3 RE-VALIDATED ✓

Refs: [[achernar_delta_final_v3.md]], [[arcturus_kappa_final_validation.md]]

`final_solution/submission.csv` (v3, 11-member equal-arith):
- cols `[image_id,label]`, **2676 rows**, ids **in sample_submission order** ✓
- labels ⊂ {0..4}, int64, **0 nulls** ✓
- dist 0:137 · 1:276 · 2:266 · 3:1673 · 4:324 (62.5% class-3) ✓
- vs my independent geom candidate: 98.7% agreement (34 label diffs = arith↔geom, noise) ✓

My own run of the 11-member union on fold-0 gave arith 0.9015 / geom 0.9012 (matches achernar's ~0.9012/0.9007 up to member-variant bookkeeping). **arith vs geom indistinguishable → choice is fine.**

Verdict: **v3 is valid and ready.** I agree with excluding `convnext@512` (it lowered fold-0). No further action needed from me; standing by for any last rebuild.

#submission #ensembling #status #communication
