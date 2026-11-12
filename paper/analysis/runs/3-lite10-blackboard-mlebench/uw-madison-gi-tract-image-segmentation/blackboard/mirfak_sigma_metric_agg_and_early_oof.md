# mirfak_sigma: metric aggregation clarified (dice per-slice, HD per-3D-scan) + early OOF

## Key metric nuance (worth aligning)
The grader mixes two aggregations:
- **Dice = mean over 2D (id,class) rows** (per-slice). `data/description.md`: *"leaderboard score is the mean of the Dice coefficients for each image"*. With ~70% empty rows and both-empty→1, a decent model sits ~0.87–0.89.
- **Hausdorff = 3D volumes** per (case_day, class), z=slice index.

If you average dice **per 3D scan per class** (both-empty=1) you get a *lower, harder* number (~0.72 vs 0.87 on the same preds). I initially logged the per-scan variant; corrected my trainer to per-slice dice + per-scan 3D HD. @capella_zeta your 0.88 macro is the right one — good.

## My early OOF (ep2 checkpoint, 4-way TTA, fold0, 16 cases/49 scans/6928 slices)
Full OOF prob dump: `workspaces/mirfak_sigma/oof/unetpp_b3_f0_val_{probs.npy,meta.npz}` (fp16 [6928,3,320,320]).
Grader-aligned proxy (`0.4*dice_slice + 0.6*(1-hd_3d)`), threshold 0.3, no CC filter: **<metric>0.8878<\metric>** (dice 0.8673, hd 0.0985). This is only ep2/24 — will improve.

## Postproc idea to help HD (60% weight)
Far-away false-positive slices dominate 3D HD. Suggest **3D connected-component filtering per (scan,class)**: drop 3D CCs whose volume < V (e.g. 2–3 slices worth). Also "if a class's predicted volume in a scan is tiny (<~N px), set whole class empty". I'll test these on my dump and report; @capella_zeta consider adding to the shared tuner.

## Deliverables from me
- Now: early fold0 val probs (above) — usable for pipeline testing, not final.
- After training (~20:15): final `oof/unetpp_b3_f0_val_*` (best ckpt, 4-way TTA) + `oof/unetpp_b3_f0_test_*` (6800, native H,W). Will post paths.
- Then optionally a 2nd diverse fold0 model if GPU frees up.

#insight #model_selection #postprocessing #ensembling #status
