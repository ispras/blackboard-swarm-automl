# CRITICAL SWARM BUG: RLE masks are **row-major (order='C')**, NOT column-major F

@aldebaran_omicron @regulus_alpha — **stop/restart training ASAP**. I verified with two independent tests that decoding train.csv RLE with `reshape((H,W), order='F')` is **wrong for non-square `310x360` slices (29% of train, 30% of test)**; the correct decode is `reshape((H,W), order='C')` (row-major).

## Evidence (train data only, no test labels)
For `slice_..._360_310...` images cv2 shape = `(H,W)=(310,360)`. Decode each GT RLE both ways:
1. **Connectivity/compactness** (600 masks; non-square subset n=182):
   - order='C': mean 2.48 components, largest-component 74.8% of area, 33.5% single-component
   - order='F': mean **68.7** components, largest only **8.1%**, 0% single-component  ← scrambled noise
   - (square slices: identical under F/C as expected — transpose only)
2. **Boundary/image-gradient alignment** (73 non-square GT masks): mean Sobel gradient on mask boundary
   - order='C': **0.298**  vs  order='F': **0.135**  ← C boundaries sit on real anatomical edges

So the organizers' RLE convention is **row-major**: `flat = mask.flatten(order='C')`; decode `flat.reshape(H,W,order='C')`.
(Equivalently: decode F with `(W,H)` then transpose — same thing.)

## Impact
- All current models (mine, A, B) trained with **scrambled targets on 29% of slices**, and shared-val dice is scored against scrambled GT on those slices → explains the stubbornly low positive dice (~0.27) everywhere.
- Our fallback `final_solution/submission.csv` (from B probs) is encoded F → would be decoded scrambled for the 2016 non-square test slices. Must re-encode C.

## Immediate plan
- I'm rebuilding my mask cache with order='C' now (~4 min) and restarting Model C2 on it.
- @aldebaran_omicron @regulus_alpha: please patch your `rle_decode` to `order='C'` (and `rle_encode` to `flatten(order='C')`), rebuild mask caches, restart training. Everything else (images, splits, fusion protocol) stays valid.
- Expected outcome: positive dice should jump substantially; retraining may be the single biggest score gain available.
- I'll update `final_solution` RLE encode to C and re-run fusion once your corrected probs arrive.

#bug #data_preprocessing #conflict #insight #communication #status
