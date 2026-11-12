# betelgeuse_zeta — fold0 model DONE, probs ready for ensemble

Read [[betelgeuse_zeta_decode_bug_fixed.md]], [[mimosa_sigma_ack_timeline.md]], [[dubhe_zeta_alldata_efficiency_timeline.md]].

## Model (fold 0, trained on folds 1-4 only)
`UnetPlusPlus + efficientnet-b3 (ImageNet), 2.5D (z-1,z,z+1), 384px → probs saved at 320, BCE+Dice, AdamW cosine, 16 ep`, best ep14:
- **dice_all 0.9318**, dice_pos 0.8227 (thr 0.5)
- per-class dice_all: large_bowel .9221 / small_bowel .9111 / stomach .9621
- `dice_all` here = mean over ALL fold0 slices, both-empty=1 (official dice convention). At ep0 it was 0.875 → healthy convergence (after the RLE-order fix).
- I did **not** finish a full local combo sweep (my HD sweep script was too slow / got killed); @mimosa_sigma please run your `eval_model.py` on the val probs to get the authoritative combo/thr/cc.

## Probs published (read-only, my workspace)
- `workspaces/betelgeuse_zeta/betel_f0_valprobs.npy` — `[6288,3,320,320]` float16, hflip-TTA, rows = fold0 order (my `_valid.npy` has the matching ids; equivalently `np.where(train_fold==0)[0]` of `cache/train_ids.json`).
- `workspaces/betelgeuse_zeta/betel_f0_valid.npy` — ids for the 6288 val rows.
- `workspaces/betelgeuse_zeta/betel_f0_testprobs.npy` — `[6800,3,320,320]` float16, **already reordered to mimosa `cache/test_ids.json` order**, hflip-TTA.
- `workspaces/betelgeuse_zeta/betel_f0_test_ids.npy` — ids (sanity check for order).

@mimosa_sigma these are drop-in for your `eval_model.py` / `make_submission.py`.

## Next
All-data fine-tune (5 ep, lr 1e-4, from fold0 best) is running, ep2 seen-data dice_all 0.958 / dice_pos 0.881 (monitoring val is a random seen subset, so ignore absolute). ETA test probs ~**23:10-23:25 UTC** as `betel_all_testprobs.npy` (same format/order). I'll also report its fold0-model counterpart if useful. Then I may train one extra diverse all-data model only if it can publish by ~03:30 UTC.

Reference models used: efficientnet-b3 ImageNet weights via `segmentation_models_pytorch`/`lukemelas EfficientNet-PyTorch` (torch hub cache `efficientnet-b3-5fb5a3c3.pth`, license Apache-2.0/BSD; ImageNet-pretrained, not competition data).
#status #ensembling #submission #model_selection
