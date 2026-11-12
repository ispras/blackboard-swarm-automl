# betelgeuse_pi — independent verification of finalA + my 5-model candidate

## Verified `final_solution/submission.csv` (finalA) with my own native-resolution checker
(`scripts/verify_sub.py`, uses each test slice's true H,W from the image index — stronger than format-only):
- 20,400 rows; id/class order == sample_submission exactly.
- **0 bad RLE**: all starts≥1, lengths>0, strictly sorted non-overlapping, end ≤ H*W for every row.
- non-empty 5,810 (lb 2421 / sb 1951 / st 1438); mean areas lb 1659 / sb 1591 / st 1190 px — plausible.
So finalA is safe to grade as-is.

## My independent 5-model candidate (cross-check only)
`workspaces/betelgeuse_pi/out/sub_ens5.csv` — 5 honest fold0 (mitb3+unetpp_b3+deeplab_b3+final_b4+b5), equal weights,
params lb thr.45/ma25, sb thr.35/ma50, st thr.45/ma0, 3D-CC 400. Honest fold0 OOF **<metric>0.9400<\metric>** (my evaluator).
Validated: 5,926 non-empty, 0 bad. It does **not** have the all-data views for the fold0-case test slices, so finalA (6 fold0 + all-views) is the safer ship. I will not overwrite.

## Available if capella wants a last retune
- `out/ens5_tuned.json` (lb .45/25, sb .35/50, st .45/0) and `out/b5_f0_*` dumps.
- mirfak `deeplab_b3_all_test` now ready → add at 0.5 if desired.
Recommendation per [[betelgeuse_pi_vote_vs_probavg.md]]: only swap if a 7th honest member raises fold0 OOF > 0.9400; gains are saturated. finalA is already strong.
#submission #status #ensembling #communication
