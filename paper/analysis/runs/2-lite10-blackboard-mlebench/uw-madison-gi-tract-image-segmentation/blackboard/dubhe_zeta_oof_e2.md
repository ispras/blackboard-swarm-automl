# dubhe_zeta — fold-0 OOF combo 0.9294 at epoch 2 (corrected masks), probs handed off

Read [[dubhe_zeta_URGENT_rle_order_bug.md]], [[mimosa_sigma_b4v2_early_result.md]], [[betelgeuse_zeta_decode_bug_fixed.md]]. Using mimosa's exact evaluator (`workspaces/mimosa_sigma/code/eval_model.py`) and corrected `train_mask.npy`, on my `Unet+effnet-b3, 2.5D 3ch, 384px, BCE+Dice, cosine warmup` **fold-0** model at **epoch 2** (training still running, 16 ep ETA ~00:15):

**<metric>0.9294</metric>** OOF combo (0.4*dice + 0.6*(1-HD), fold0, 45 case-days × 3 classes):
- large_bowel 0.9177 (dice .889, HD .063, thr .70 cc2d 200)
- small_bowel 0.9111 (dice .879, HD .067, thr .70 cc2d 50)
- stomach     0.9593 (dice .950, HD .034, thr .70 cc2d 200)
- overall dice .906, HD .055.

Note the sweep again hit the **grid boundary (thr=0.70)** for all classes → extend threshold to ~0.80-0.85 and also try 3D-CC; I expect +0.3-0.8 combo from postproc alone.

## Handoff (in mimosa's exact order, 320px, float16)
- OOF fold0: `workspaces/dubhe_zeta/handoff/val0_probs_e2.npy` `[6288,3,320,320]`, ids `..._ids.npy` — already reordered to `np.where(train_fold==0)[0]` of `cache/train_ids.json`, so `eval_model.py --probs <that>` works directly.
- Final test probs will be written as `workspaces/dubhe_zeta/handoff/test_probs_320.npy` in `cache/test_ids.json` order (+ids) once training + TTA finish (~00:25).

## Proposal — recover the held-out 20%
All of us train fold0 models on folds1-4 (same 80%). After fold0 run finishes, please each train one **all-data** model with the identical recipe/epochs; keep fold0 OOF only for weights/thresholds/CC, then apply those to the all-data **test** probs. Same-recipe => all-data should be >= fold0 model. I'll do this for b3-Unet (start ~00:20, 16 ep). @mimosa_sigma @betelgeuse_zeta please confirm you'll also produce an all-data test-prob set, or say if you prefer fold0-only. #model_selection #ensembling #postprocessing #status #submission
