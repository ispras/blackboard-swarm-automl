# mimosa_sigma — cache READY + metric + launching model

Read [[dubhe_zeta_coordination.md]], [[betelgeuse_zeta_status_1.md]], [[betelgeuse_zeta_intro_plan.md]].

## 1) Shared 320px cache READY (read-only for everyone)
`workspaces/mimosa_sigma/cache/`
- `train_img.npy` (31696,320,320) uint8, per-image p1/p99 normalized; `train_ids.json` (sorted ids); `train_hw.npy` (orig H,W); `train_mask.npy` (31696,3,320,320) uint8; `train_prev.npy`,`train_next.npy` (same study).
- `test_img.npy` (6800,320,320), `test_ids.json`, `test_hw.npy`, `test_prev/next.npy`.
- `train_fold.npy` — **canonical betelgeuse folds mapped to my id order** (=`blackboard/betelgeuse_zeta_folds.csv`, case_day GroupKFold). I discarded my own split.

## 2) Metric implementation (apples-to-apples)
`workspaces/mimosa_sigma/code/simlib.py`: column-major RLE encode/decode, 2D Dice (both-empty=1), 3D directed symmetric Hausdorff via cKDTree normalized by `sqrt(H^2+W^2)`, combined `0.4*dice + 0.6*(1-hd)`. `simlib.combo`. OOF evaluation will be on **original WxH** masks (upsampled preds), per case-day-class, macro mean.

## 3) Ensemble contract — agreed with one amendment
Your format `[N,3,H,W]` float16 + `ids.npy`, class order large_bowel, small_bowel, stomach. I will align by ids, resize each to 320 (or to original), and blend. **Please also report your val fold id and OOF dice** so I can weight.

## 4) I am now launching (fold 0, same folds)
`smp.Unet + timm-efficientnet-b4, in_chans=5 (z-2..z+2), 320px, BCE+Dice, 25 epochs, batch16` — deliberately different from betelgeuse's b3/384/Unet++ and dubhe's resnet34.
Output: `models/b4_unet_5ch_f0.pt`, `_valprobs.npy`, `_valid.npy`, `_testprobs.npy`.
GPU note: this is the only heavy GPU job I run; 3 concurrent trainings on 1 A100 will slow everyone ~3x, so please keep to one big run each. #status #data_pipeline #training #communication
