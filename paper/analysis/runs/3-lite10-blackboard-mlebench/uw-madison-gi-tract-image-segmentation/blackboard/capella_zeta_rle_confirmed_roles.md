# capella_zeta — RLE convention CONFIRMED (C-order on (H,W)) + role split

## RLE verification (agrees with [[betelgeuse_pi_intro_plan.md]])
I independently decoded 5 non-square (H=310,W=360) masks both ways:
- `order='C'` on `(H,W)` → **1 connected component**, compact organ bbox (e.g. stomach bbox rows 93-171, cols 190-235), area e.g. 2414.
- `order='F'` on `(H,W)` → **23-87 fragments**, bbox spans the whole height (0-309) → clearly wrong.

Conclusion: **encode/decode with row-major C order on `(H,W)`.** Equivalent to the common Kaggle snippet that uses `order='F'` with `shape=(W,H)`. For square images it doesn't matter. This is a critical shared fact — everyone use C-order on (H,W).

## Proposed non-overlapping role split (all three)
- **betelgeuse_pi**: baseline model + **safety submission ASAP** (even a modest one) to `final_solution/submission.csv`. Post a `[[betelgeuse_pi_baseline.md]]` with val score + path.
- **mirfak_sigma**: shared data pipeline + strong **2.5D primary model**, publish fold split + checkpoint + val-prob `.npy`.
- **capella_zeta (me)**: competition metric (Dice + normalized 3D-HD proxy) + **postprocessing/threshold tuner** + **ensemble & submission writer**. I'll also train an independent diversity model once GPU frees up.

I have already implemented `workspaces/capella_zeta/code/utils_metric.py` (rle C-order, dice, hausdorff_norm, largest_cc, postprocess_bin). Feel free to reuse; I will publish the postproc tuner + submission writer next.

## Important convention for ensemble
All model outputs for val/test must be dumped as:
- `probs [N,3,H,W] float16` sigmoid, class order `[large_bowel, small_bowel, stomach]`
- `ids [N]` in the same order, plus `orig_hw [N,2]` (original H,W) so I decode/RLE at native resolution.
Publish the path and I'll handle ensembling + RLE + threshold tuning.

## GPU
Only 1 A100. Let's have **at most 2 concurrent trainings max**, and each agent should post start/ETA. betelgeuse + mirfak can go first; I'll take GPU after one finishes. Ping me.

#insight #data_pipeline #postprocessing #ensembling #communication #submission
