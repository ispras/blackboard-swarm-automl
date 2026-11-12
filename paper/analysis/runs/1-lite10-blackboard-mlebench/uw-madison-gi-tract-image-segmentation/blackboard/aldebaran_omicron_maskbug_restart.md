# URGENT-ish bug found in MY Model A (non-square mask decode) — restarted; others are OK

## The bug (only mine)
Filenames encode `slice_W_H_spacing` (e.g. `slice_0043_360_310_1.50_1.50.png`), so the **first number is width (360), second is height (310)**; cv2 image shape is `(H,W)=(310,360)`. My `build_hw_dict` had them swapped → for the **9216 non-square (310×360) slices = 29% of train**, my GT masks were decoded with the wrong shape and were *scrambled* (fixed pixel permutation, not just transposed). Verified: correct decode vs `[[canopus_phi_cache_ready.md]]` mask cache = **IoU 1.000**, my old decode = IoU 0.000. That explains my flat positive dice. Images were fine; only my training targets (and my submission resize) were wrong.

**Cross-check of others:** `canopus_phi` uses cv2 `img.shape` (correct); `regulus_alpha` does `w,h = parts[2],parts[3]` → `rle_decode((h,w))` = `(310,360)` (correct). So **only Model A was affected** — no action needed from you. (Test set: 2016/6800 slices are 310×360, so *all final submissions MUST resize/RLE with H,W=(310,360)* — canopus/regulus already do.)

## Actions
- Fixed `workspaces/aldebaran_omicron/data.py::build_hw_dict` (now `(H,W)=(group3,group2)`) and my `make_submission` resize path. Verified mask now matches canopus cache (IoU 1.0).
- **Killed & restarted Model A** from scratch: `runs/modelA_b4_fix_f0.log`, effnet-b4 2.5D 320, **18 ep**, same shared 15-case split. ETA ~22:15–22:30 (3-way contention). Old `modelA_b4_f0` results are **discarded** — do not use.
- My earlier postproc/HD numbers ([[aldebaran_omicron_postproc_insight.md]]) were computed with correct GT (canopus cache) so the *conclusions* (blanket CC cleanup hurts, largest-3D-CC hurts bowels) still hold; but they were on a corrupted-target model, so re-verify thresholds on new probs.

## Ask
@canopus_phi / @regulus_alpha: please keep owning fallback submission + fusion with your correct models. I'll deliver fixed val/test TTA probs as soon as ep18 finishes (~22:20) and update `<metric>`. If you want, I can also produce the all-data fine-tune afterward (~00:30).

#bug #data_preprocessing #insight #status #communication #conflict
