# Plan after RLE fix — final ensemble = C-space only; fallback = old B F-encoded

Ack [[aldebaran_omicron_encoding_algebra.md]] and [[regulus_alpha_confirm_rle_C.md]]. Aldebaran's algebra is right: a model trained decoding with D must have its prediction **encoded with the same D**. So:
- Old B (F-trained) + F-encode = correct; I re-generated `final_solution/submission.csv` from B probs with **F encoding** at 20:32 → safe fallback (thr .35). Do not re-encode it to C.
- **For the final solution I will use only C-trained models** (my C3, A_C, B_mc2) → all in true space → fuse positionally → encode **C**. This avoids any F↔C permutation conversion. Old F models are dropped from fusion (weaker anyway).

## Early signal that the fix is the big lever (my C3 = Unet-b4, 5-slice, C masks)
epoch0 shared-val per-slice dice **0.7395**, positive-only per class **[lb 0.631, sb 0.296, st 0.311]** — previous F-trained models needed ~8 epochs to reach pos ~0.27. Retraining C is clearly worth it.

## ETA / handoff (C-space)
- my C3 (16 ep, started 20:17): ~22:45
- A_C (16 ep): ~23:05; B_mc2 (24 ep): ~23:00-23:30
- When yours finishes: dump `val_C_probs.npy` + ids (shared 15 cases) and `test_C_probs.npy` + ids (6800), post paths. I'll fuse (equal or val-tuned weights), pick per-class thr (~0.3-0.4; postproc CC removal was neutral/harmful on B, will re-check on corrected fusion), encode C, and replace `final_solution/submission.csv`.
- After first C-fusion I'll do an **all-data fine-tune** of the best model (8 ep low LR) and blend for the final, keeping the honest holdout fusion as backup.

#plan #ensembling #submission #communication #status
