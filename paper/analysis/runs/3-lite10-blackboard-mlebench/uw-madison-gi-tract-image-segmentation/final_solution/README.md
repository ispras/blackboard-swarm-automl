# UW-Madison GI Tract Image Segmentation — final solution

Swarm solution (agents: `capella_zeta`, `mirfak_sigma`, `betelgeuse_pi`).
Task: 2D MRI slice segmentation of 3 classes (`large_bowel`, `small_bowel`, `stomach`);
metric `dice-hausdorff-combo` = 0.4·Dice(per-slice) + 0.6·(1 − normalized 3D Hausdorff per scan).

## Result
- `submission.csv`: 20,400 rows (`id,class,predicted`), RLE (row-major / C order on `(H,W)`, 1-indexed).
- Honest fold-0 (case-level holdout, 16 cases / 6928 slices) proxy for the **6-model fold-0 ensemble**:
  **0.4·dice + 0.6·(1−HD3D) = 0.9396** (dice 0.9163, HD3D 0.0448).
  Metric convention: empty-vs-empty treated as perfect (dice 1, HD 0).
- Progression: best single model 0.9359 → 3-model 0.9381 → 6-model 0.9394 → refined postproc 0.9396.
  A 7th honest model (Unet++/b4) was evaluated and gave no OOF gain (0.9160 vs 0.9158 at thr 0.45), so it was excluded.

## Pipeline
1. **Preprocessing** (`preprocess.py`): 16-bit PNG → per-slice 1st/99.5th percentile clip & scale to uint8 →
   resize 320×320 (image INTER_AREA, mask INTER_NEAREST). RLE decoded with **C order on true (H,W)**.
   2.5D input: 3 adjacent slices (edge-clamped) as 3 channels (`z-1,z,z+1`); one 5-slice model (z-2..z+2).
2. **Models** (all ImageNet-pretrained encoders, SMP 0.3.4 / timm 0.9.7, no competition-specific weights):
   - `mit_b3` SegFormer-B3 + Unet, 24 ep, fold0 (capella_zeta)
   - `efficientnet-b4` + Unet, **5-slice** input, 16 ep, fold0 (capella_zeta)
   - `efficientnet-b3` + UnetPlusPlus, 24 ep, fold0 (mirfak_sigma)
   - `efficientnet-b3` + DeepLabV3Plus (ASPP), 20 ep, fold0 (mirfak_sigma)
   - `efficientnet-b4` + Unet, 36 ep, fold0, 3-slice (betelgeuse_pi)
   - `efficientnet-b5` + Unet, fold0 (betelgeuse_pi)
   Training: 0.5·BCE + 0.5·soft-Dice, AdamW + OneCycle, AMP, bs 20–32, flips/affine/brightness aug,
   case-level 5-fold split (seed42 `i%5`), fold0 = validation.
   Four full-data fine-tunes (8 ep, all 76 cases, lr 1e-4/5e-5) from the mit_b3, Unet++/b3, b4 and
   DeepLab/b3 fold-0 checkpoints produce extra test-only views (they cover the 864 test slices whose
   case is held out in fold0).
3. **Inference** (`infer.py`): 4-way flip TTA (identity+h+v+hv), sigmoid probs dumped as
   fp16 `[N,3,320,320]` + ids with native `H,W`.
4. **Ensembling** (`finalize.py`): id-aligned weighted average of sigmoid probs
   (6 fold0 models weight 1, four all-data views weight 0.5 for the test blend).
5. **3D postprocessing + submission** (`tune3d.py`): per-class threshold (lb .55 / sb .55 / st .55),
   per-slice hole fill, 3D connected-component removal (`min_vol3d` lb 1200 / sb 200 / st 2000, 26-conn),
   per-scan empty rule (lb 300); threshold/volume grid tuned on fold0 (per-slice Dice + per-scan 3D HD).
   Final params in `code/tuned3d_params_final.json` (submission.csv = `run_finalB.sh` test blend + these params).
   Masks resized to native `(H,W)` then RLE-encoded row-major (`mask.reshape(-1)`).

## Provenance of external models (allowed general ImageNet backbones only)
| model | source | purpose |
|---|---|---|
| efficientnet_b3 / b4 / b5 | `timm` / `segmentation_models_pytorch` ImageNet weights | segmentation encoders |
| mit_b3 (SegFormer-B3) | `segmentation_models_pytorch` (ImageNet) | segmentation encoder |
No pretrained checkpoint was trained on this competition's labels; no competition solutions/write-ups were used.

## Reproduce
```
# training (example)
python code/train.py --arch unet --encoder mit_b3 --n_slices 3 --fold 0 --epochs 24 --bs 24 --lr 3e-4 --tag mitb3_f0
# inference
python code/infer.py --ckpt ckpt/mitb3_f0_best.pt --split val --fold 0 --tag mit_val --tta h,v
python code/infer.py --ckpt ckpt/mitb3_f0_best.pt --split test --tag mit_test --tta h,v
# ensemble + 3D postproc + submission (see specs_test_10.json / run_finalB.sh)
```
The shipped CSV `submission.csv` was produced by `run_finalB.sh` (6 fold0 + 4 all-data test views,
postproc params from `tune3d.py` on the 6-model fold0 average). `run_finalA.sh` is the 6+3 variant.

Teammate probability dumps are referenced in `specs_test_9.json`; each is produced by the same
convention (fp16 sigmoid, class order [large_bowel, small_bowel, stomach], native H,W).

Board messages for provenance: `capella_zeta_finalB_shipped.md`, `mirfak_sigma_deeplab_final_adds_gain.md`,
`betelgeuse_pi_final_probs_ready.md`, `capella_zeta_ens3_and_allviews.md`, `capella_zeta_rle_resolved.md`.
