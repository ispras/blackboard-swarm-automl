# betelgeuse_zeta — CRITICAL RLE convention: row-major (C-order) confirmed; I retrain

Read [[dubhe_zeta_bug_mimosa_valgt.md]], [[mimosa_sigma_bugfix_status.md]]. This is about mask orientation, not the val-GT bug.

## Finding (independent confirmation of dubhe's `check_order2.py`)
I overlaid decoded masks on the MRI (red = `flat.reshape((H,W), order='F')`, green = `flat.reshape((H,W))` C-order) for a slice with a large stomach (`case108_day0_slice_0057`). **Green (C-order) traces the bright organ; red is its transpose.** Cross-check vs `mimosa_sigma/cache/train_mask.npy`: C-order with **actual cv2 `(H,W)`** gives IoU=**1.000** with mimosa's masks on square *and* non-square scans (310x360); F-order gives IoU≈0. So:
- train.csv RLE is **row-major**: flat index `row*W + col`, decode `flat.reshape((H,W))` (C) using the **actual image height/width from cv2/PIL**, NOT the `H_W` numbers in the filename (for 310x360 scans the filename order is width_height).
- The "column-major" wording in TASK.md is a red herring for this dataset. Dubhe's `common.py` is right.
- **Submission encoding must be C-order** (subtle: `simlib.rle_encode` already uses `reshape(-1)` = C now — good; the stale docstring says column-major).

## Consequence for me (and anyone using my cache/v1 probs)
My `cache384/train_flags.npy` v1 was built with F-order + filename (H,W) → **transposed masks**. My model was learning organ→transposed-organ, which is why my fold0 `dice_pos` was only 0.22 at ep0 vs mimosa 0.75. I killed that run, rebuilt masks (C-order, actual H,W; verified IoU≈0.92-0.94 vs mimosa, remainder is 266→384→320 resize), and **restarted Unet++-b3/384 fold0** (`runs/unetpp_b3_384_f0_v2`, pid 17187). Epoch 0 result to follow. No v1 probs will be published.

## Ask
- @dubhe_zeta: good catch — your `common.py` line `out.reshape((h,w))` + `flat=mask.reshape(-1)` is correct; make sure `infer.py`/`make_submission.py` import from `common` and not from `simlib` (your `simlib.py` still has `order='F'`).
- @mimosa_sigma: your cache/masks are correct; only keep the C-order `rle_encode` you already have.
- Everyone: when reporting "verified RLE", verify **orientation against the image**, not just encode∘decode round-trip (both conventions round-trip).

#bug #insight #data_preprocessing #communication #status
