# mimosa_sigma — b4v2 after C-order mask fix: OOF combo 0.9123 at epoch 1 (!)

Read [[mimosa_sigma_rle_confirmed.md]], [[dubhe_zeta_URGENT_rle_order_bug.md]].

## Big confirmation that the RLE fix was the whole problem
Same architecture/recipe, only masks corrected (C-order). Fold-0 val after **1 epoch**:
- old (F-order masks): `dice_all 0.728`, `dice_pos 0.287`
- new (C-order masks): `dice_all 0.864`, `dice_pos 0.698` per class .75/.65/.69

## OOF metric with postproc (my `eval_model.py`, full 3D-HD, fold 0, 45 case-days × 3 classes)
`Unet+effnet-b4, 5ch 2.5D, 320, epoch 1`, threshold 0.6 + 2D CC min-area 200:
<metric>0.9123</metric> (combo = 0.4*dice + 0.6*(1-HD); dice_all 0.876, HD 0.064)

Per class (thr 0.6, cc2d 200): large_bowel 0.900 / small_bowel 0.901 / stomach 0.936.
The sweep hit the **grid boundary** (thr 0.6, cc 200) for every class, so higher thr / stronger CC still helps -> I'll extend to thr≤0.8 and cc≤1000 + 3D-CC at the end.

Notes:
- This is OOF on fold 0 only (not test). Model training continues (20 ep, ETA ~00:15).
- HD weight 0.6 ⇒ aggressive FP suppression (high thr + CC) is the main postproc lever; empty/false-positive slices dominate HD.
- @dubhe_zeta @betelgeuse_zeta: expected ensemble gains are in the last ~1-2% combo. Once your corrected models save fold-0 OOF + test probs, I'll blend and tune per-class thr/CC jointly (and validate the tuning on a held-out half of fold-0 groups to avoid overfitting thresholds).

Reference: [[betelgeuse_zeta_status_2.md]] [[dubhe_zeta_status_2.md]] #insight #model_selection #postprocessing #status #ensembling
