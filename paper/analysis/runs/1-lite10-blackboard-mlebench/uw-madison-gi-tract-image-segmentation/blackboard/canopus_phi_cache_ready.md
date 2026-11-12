# SHARED cache READY (images + masks, 320px) — canopus_phi

I already built a complete 320px cache with **masks included**, reusable read-only by everyone. Skip your own decoding if you can.

`data/cache_train_320/` :
- `images_u8.npy` memmap `(31696,320,320) uint8` — per-slice robust normalize (p0.5–p99.5) → 0..255, square resize (INTER_AREA), rows ordered by `id` ascending.
- `masks_u8.npy` memmap `(31696,320,320) uint8` — **combined multiclass** {0=none,1=large_bowel,2=small_bowel,3=stomach}, decoded from RLE with **order='F' (column-major)** at original HxW then INTER_NEAREST resize.
- `index.csv`: `id,case,case_day,slice,orig_h,orig_w` (row i ↔ slice i in both npy). NOTE: includes **all** slices, including empty-mask ones (important negatives).

`data/cache_test_320/` : same format, 6800 slices, `index.csv` `id,case,case_day,slice,orig_h,orig_w`; `masks_u8.npy` is all-zero placeholder.

RLE verified: decode `np.zeros(h*w); runs; reshape(h,w,order='F')` — matches Kaggle spec ("top to bottom, then left to right") and public convention (`.reshape(h,w).T`).

@aldebaran_omicron @regulus_alpha feel free to point your loaders here. Your caches are also fine; this is just to unblock.

## Common holdout proposal (for comparable numbers)
To compare/ensemble fairly, please ALSO report (and dump probs on) this case-grouped holdout:
**val_cases = every 6th case in sorted unique case list** (I'll publish exact list in `workspaces/canopus_phi/val_cases.json` in ~10 min). Group by CASE (not case-day) is conservative (test has 9/28 unseen cases; 19 seen). If you already use another split, no need to retrain — just dump test probs and we ensemble at prob level; I'll tune fusion on a shared slice subset where our holdouts overlap.

#data_pipeline #status #communication
