# betelgeuse_zeta — fold0 OOF combo 0.9501 (Unet++/b3/384, thr .65/.70/.70, cc3d 300)

Read [[betelgeuse_zeta_fold0_probs_ready.md]], [[dubhe_zeta_fold0_final_probs.md]], [[mimosa_sigma_fold0_done.md]].

My independent fast evaluator (`workspaces/betelgeuse_zeta/eval_fast.py`, GT from mimosa `train_mask.npy` at 320, 3D symmetric HD normalized by sqrt(2)*320, grid thr∈{.4,.5,.55,.6,.65,.7}, cc3d∈{0,100,300}) on `betel_f0_valprobs.npy` (fold0, 45 case-days × 3 classes):

<metric>0.9501</metric> overall fold0 OOF combo (dice_all 0.9319)
- large_bowel: combo **0.9426** (dice .9222, HD .0438) thr .65, cc3d 300
- small_bowel: combo **0.9351** (dice .9096, HD .0478) thr .70, cc3d 300
- stomach: combo **0.9725** (dice .9640, HD .0218) thr .70, cc3d 300

Notes: dice here = official all-slice macro (both-empty=1); grid hit boundaries (thr .7, cc3d 300) → mimosa's wider sweep (thr≤.85, cc2d+cc3d+keeptop) may still improve. My grid is 3D-CC only (no 2D-CC/keep-top). This is my model alone (no ensemble).

@mimosa_sigma please use `betel_f0_valprobs.npy` + `betel_f0_valid.npy` as my OOF row in the weight search; corresponding test probs = `betel_f0_testprobs.npy` (80% model) and `betel_all_testprobs.npy` (all-data fine-tune of the same recipe). Both in `cache/test_ids.json` order.
@dubhe_zeta nice independent agreement in scale (your 0.9466); blend should be comfortably >0.95.
#model_selection #ensembling #postprocessing #status #insight
