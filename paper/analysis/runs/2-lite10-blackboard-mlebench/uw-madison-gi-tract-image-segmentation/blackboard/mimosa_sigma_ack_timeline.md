# mimosa_sigma — ACK all-data fine-tune plan + my deliverables/timeline

Read [[dubhe_zeta_alldata_efficiency_timeline.md]], [[betelgeuse_zeta_confirm_alldata.md]]. **ACK** the fine-tune-from-fold0-checkpoint plan and the **05:30-06:30 MSK final freeze**. Good call on fine-tune vs retrain.

## My status
- b4v2 fold0 (Unet/effnet-b4/5ch/320) training, currently ep6 `dice_all 0.917` `dice_pos 0.766`; ETA ep20 ~**00:15 MSK**.
- Added `--all-data 1 --init-ckpt <fold0 best.pt>` to my `train.py`. Will run **5 epochs @ lr1e-4** on all 225 case-days from the fold0 best ckpt, ETA ~**00:55 MSK**. Then TTA test probs.
- RLE now proven exactly: `simlib.rle_encode(simlib.rle_decode(train.csv_row))` reproduces the original train.csv RLE **string-exactly** for both 266² and 310x360 — independent of the image-alignment test. Submission encoder is safe.
- `tune_postproc.py` ready: per-class staged grid (thr 0.45-0.85, cc2d, cc3d, keep-top), tuned on **half the fold-0 case-days**, reported on the other half (honest OOF). `make_submission.py` accepts the recipe JSON.

## Deliverables (paths I will publish)
- `workspaces/mimosa_sigma/models/b4v2_f0_valprobs.npy` + `_testprobs.npy` (fold0 model, hflip TTA, 320, `cache/test_ids.json` order)
- `workspaces/mimosa_sigma/models/b4v2_all_f0_testprobs.npy` (all-data fine-tuned, TTA)
- `workspaces/mimosa_sigma/logs/recipe_<tag>.json` + `workspaces/mimosa_sigma/ensemble/submission.csv`

## Ask
Publish each model's probs as `[N,3,320,320]` float16 in **`cache/test_ids.json` order** (test) and **fold0 `np.where(train_fold==0)[0]` order** (OOF val). Name them clearly with model+fold0/all. Tell me per-class OOF combo if you compute one. I'll blend fold0 OOF probs, tune weights+thr/CC on half the groups, validate on the other half, then apply the recipe+weights to the all-data test probs.
If your fold0 OOF probs and all-data test probs use different recipes, say which. #plan #ensembling #communication #status
