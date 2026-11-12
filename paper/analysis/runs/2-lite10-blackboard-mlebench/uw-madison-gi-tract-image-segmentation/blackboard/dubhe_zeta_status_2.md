# dubhe_zeta status 2 — baseline training + RLE in place, final_solution ownership

Read [[mimosa_sigma_cache_metric_ready.md]], [[betelgeuse_zeta_status_1.md]].

## Progress
- `smp.Unet + resnet34, 2.5D(z-1,z,z+1), 320px, BCE+Dice, OneCycle`, fold 0 of [[betelgeuse_zeta_folds.csv]] (I load the board CSV directly, so folds are canonical). ~4 min/epoch under 2-3-way GPU sharing; 30 epochs ETA ~22:55.
- Verified **RLE column-major** independently; my `rle_encode`/`rle_decode` and [[mimosa_sigma_cache_metric_ready.md]] `simlib.py` agree (`reshape(h,w,order='F')` == my `reshape(w,h).T`). So submissions are format-correct.
- Have `infer.py` (TTA, saves `[N,3,320,320] float16 probs + ids + gt + meta`) and `make_submission.py` (per-class thr, 3x3 close, min-area CC filter, upsample→orig WxH, RLE). Will publish test probs + a safety-net submission as soon as val is non-trivial (~epoch 10).

## Ownership proposal to avoid write conflicts on `final_solution/`
**I will be the single writer of `final_solution/`** (submission.csv + code + README with external-model attribution). `mimosa_sigma` sends me the tuned ensemble recipe/weights/thresholds + the probs; `betelgeuse_zeta` sends probs. I copy the winning candidate and never touch others' files. If mimosa prefers to own the final write, reply and I'll hand it over — just make sure only one of us writes there.
- @betelgeuse_zeta: no training process visible on the box yet (~20 min). Status? If you're blocked on cache/code, say so; I can share my dataset/infer code. We need your probs for ensemble diversity.
- @mimosa_sigma: when you have OOF eval on val fold 0, publish per-class dice + combo so we can compare model quality early. Also tell me the exact npy naming/location you want my test probs at (I'll write into `workspaces/dubhe_zeta/` read-only for you).

Note: mean val dice (positive slices) after 2 epochs is only ~0.17 (early); will update with real numbers. #status #inference #submission #communication
