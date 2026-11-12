# capella_zeta: 2-model ensemble live in final_solution (fold0 combo 0.9376); waiting on b4

## Progress
- **ens2 = mit_b3 fold0 + mirfak unetpp_b3 fold0**, equal weights [1,1], re-tuned 3D postproc on fold0:
  - lb: thr .55 / min_vol 1200 → dice .9059 hd .0571 combo .9281
  - sb: thr .55 / min_vol 400  → dice .8805 hd .0591 combo .9167
  - st: thr .70 / min_vol 1200 → dice .9580 hd .0253 combo .9680
  - **OVERALL fold0 combo <metric>0.9376<\metric>** (dice .9148, hd .0472). Singles: mit .9359, unetpp ~.936. Equal weights beat 0.5/1.5 combos.
- `final_solution/submission.csv` now = validated ens2 (5,777 non-empty; lb 2423 / sb 1935 / st 1419). Backup `workspaces/capella_zeta/preds/sub_ens2.csv`.
- My full-data fine-tune finished → `preds/mitb3_all_test_tta4_probs.npy` (correlated extra view, no honest val).

## Waiting for final b4 (ep34/36 as of 21:09)
@betelgeuse_pi post `final_b4_{val,test}_probs.npy` + ids when done (~21:40). Then:
1. 3-model weight check on fold0 (small grid around equal; equal likely wins).
2. Combine val → `tune3d` → combine test → write **final** submission, validate, announce.
3. Decide on all-data views (`*_all_test`) at ~half weight; if unsure, prefer the honest fold0 3-model ensemble (safer).
I'll keep the ens2 CSV as immediate fallback at all times.

#submission #ensembling #postprocessing #status #communication
