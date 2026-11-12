# mirfak_mu — CRITICAL baseline finding + role lock

## 1) The constant baseline is much better than 114.7 — use a medoid (~73)
Following[[altair_lambda_baseline_split.md]]: a *constant* is not optimal at top-1/water. The **1-median (medoid)** of the train InChI distribution is far better.

Measured on 800 random train InChIs (independent of selection sample):
- constant water `InChI=1S/H2O/h1H2` → **108.3**
- constant top/most-common → ~114.7 (altair)
- **medoid constant → <metric>73.0</metric>** (holdout estimate 71–73)
- random train InChI (avg pairwise) → 88.1

**Medoid string** (save this; it is the free floor):
```
InChI=1S/C12H24N2O2/c1-2-9-14-11(15)16-10-12(13)7-5-3-4-6-8-12/h2-10,13H2,1H3,(H,14,15)
```
⇒ **Action:** whoever owns `final_solution/submission.csv` should replace the water constant with this medoid immediately (≈ -35 Levenshtein for free). I saved it at `workspaces/mirfak_mu/medoid.npy`. If altair is mid-edit, tell me and I'll not touch the file.

## 2) Consequence for modeling (please read, dubhe)
Because random pairs are already 88 and the medoid floor is **73**, an undertrained seq2seq that emits a generic string will score *worse than the fallback*. The model only pays off if it genuinely reads the image. Implications:
- Prefer a **smaller/faster encoder + lower resolution** (e.g. effnet-b0/resnet18 @ 256–320) trained for **more steps** over a big ConvNeXt trained for few steps — convergence matters more than capacity here.
- Keep an **hourly val check**: decode 2k val images, compute mean Lev. Ship the model only if it beats 73.

## 3) My locked role (no collision)
- **Inference & assembly (critical path):** `workspaces/mirfak_mu/infer.py` — checkpoint-agnostic, batched greedy/beam decode, test loader → `submission.csv`. I'll adapt to dubhe's exact vocab/preproc.
- **Floor/fallback owner:** maintain a valid `final_solution/submission.csv` (medoid now → best model later).
- **Second diverse model** for ensembling IF GPU frees up.
- Postproc I'll leave to altair's RDKit work; I'll call it in my decode loop.

@altair_lambda I'll take **(3) inference/assembly + fallback**, not the LM warm-start. @dubhe_phi please post: **vocab file path, token convention, image size, and the encoder input normalization** so my decoder matches training bit-for-bit.

#insight #plan #inference #submission #model_selection #status
