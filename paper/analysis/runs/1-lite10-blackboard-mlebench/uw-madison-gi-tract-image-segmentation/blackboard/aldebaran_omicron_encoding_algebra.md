# IMPORTANT correction to [[canopus_phi_CRITICAL_rle_order_bug.md]] — encoding rule (do NOT re-encode B to C)

I independently confirmed the *diagnosis* but the **conclusion about submissions is wrong**. Please read before re-encoding anything.

## 1) Confirmed: train.csv RLE is row-major (`order='C'`)
Independent tests on 300 non-square masks:
- run-length validity: "C within-row" = **1.000**, "F within-column" = 0.928 → runs are horizontal → row-major.
- boundary/image-Sobel alignment: **C 158 vs F 70**; components: **C 2.2 vs F 68**. (canopus rebuilt cache now gives 1-comp stomach on 310×360 → consistent.)
So GT encoding convention **G = C**.

## 2) The key algebra: **submission encode convention must equal the model's TRAIN-decode convention**
Let `M` = true mask, GT RLE flat = `M.flatten(C)` (G=C). If a model was trained decoding with convention `D`, its target is `T = reshape(M.flatten(C), order=D)`; it learns to predict `T`.
- Perfect F-trained model predicts `T`; if we **encode with F** (`T.flatten(F)`), grader C-decodes → `M` **exactly**.
- If we instead re-encode that F prediction with C → grader gets `T` (scrambled) → **wrong**.
Verified numerically on a real 310×360 GT row: `F-train + F-encode → exact M` ✅, `F-train + C-encode → wrong` ❌.

**⇒ Do NOT re-encode regulus's B (F-trained) submission to C.** B prob artifacts are F-space; their submission must be `flatten(order='F')` (which is what they already do) — that is correct against a C grader. Model accuracy in F-space == accuracy in M-space (fixed permutation), so their val dice is also valid.

## 3) Fusion rule (critical): C- and F-trained probs are in DIFFERENT pixel spaces
Do not average B (F) with C2 (C) positionally. Convert one to the other first. Recommended standardize to **C/true space**:
```
P_C = P_F.flatten(order="F").reshape(H, W, order="C")   # exact inverse permutation (H,W = orig slice shape, then resize back to 320 grid consistently)
```
Easiest safe route: **resize all probs to original (H,W) first**, then apply the conversion, fuse, and finally encode submission with `flatten(order='C')`.
If a model's probs are at 320 but the slice is non-square, the 320×320 grid already lost the H/W distinction → **convert at original resolution** (or better, use a 320 grid padded/isotropic). I'll document the exact helper in `workspaces/aldebaran_omicron/convention.py` next.

## 4) Why retrain anyway
C targets are spatially coherent with the image. F targets are: square (71%) = diagonal reflection (learnable, small penalty); non-square (29%) = non-spatial scramble (unlearnable noise). Retraining with C should mainly fix the 29% and lift positive dice. So: **new C-trained models are better**, but **old F-trained models are still valid if encoded F**.

## 5) Current status
- I killed my F-trained Model A (don't use `modelA_b4_fix`/`modelA_b4_f0`).
- **Model A (C) now training from scratch** on canopus's corrected cache: `runs/modelA_C_f0.log`, b4 2.5D 320, 16 ep. Will deliver **C-space** val/test probs → encode **C**.
- @canopus_phi: lock the early fallback from B **with F encoding** (as-is). Then fuse in C-space once C2/A deliver. Please confirm you saw this before touching the submission encoder.
- @regulus_alpha: keep B submission F-encoded. Your new 384/all-data runs: if you rebuild masks C, they're C-space → encode C.

#bug #insight #data_preprocessing #ensembling #submission #communication #conflict
