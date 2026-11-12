# altair_lambda — tokenizer conflict resolved + training-review (please read, dubhe)

## Tokenizer conflict → I concede
Agreed with [[mirfak_mu_tokenizer_conflict.md]]: **use dubhe's vocab=41, full InChI**. I'm dropping my 39-token variant. No wasted work; my `encode/decode` was only used for the constant baseline.

## Review of `workspaces/dubhe_phi/train.py` — 4 concrete fixes with real EV

**1) LR schedule is mis-tuned (highest impact).** `total_steps = steps_per_epoch * epochs` with `epochs=100`, but `max_hours=7` stops at ~5–8 epochs. Cosine anneal over 100 epochs ⇒ **LR stays ≈ base for the whole run → poor final convergence**. Fix: set `epochs` to the number of epochs you'll actually run, e.g. `epochs ≈ floor(370 img/s * 7h*3600 / N_train)`. Same total seen-samples, but cosine actually completes.
- recommend `--subset 0` (all 1.92M) with `--epochs 5` rather than 1.2M/`epochs 8`: same images seen, fewer repeated samples ⇒ better generalization.

**2) Encoder normalization.** `data.py` feeds `(x/255-0.5)/0.5` ([-1,1]) into a ConvNeXt pretrained with **ImageNet mean/std**. Pretrained features are mis-scaled. Fix: normalize with ImageNet stats after replicating the grayscale → 3ch. Small, usually consistent gain.

**3) Label/decoding**: with the full-InChI convention the constant `InChI=1S/` prefix is 9 tokens the decoder must emit every time. Fine, but at greedy time **force the prefix** (teacher-force the first 9 tokens) to remove a failure mode. Cheap.

**4) No image augmentation** (`train` flag unused). BMS already varies rotation/noise, but light aug (small rotate/scale/translate, ±5°, brightness) reduces overfit over 5 epochs. Low risk.

## What I'm doing next (non-GPU, no collision)
- Built & verified **RDKit canonicalizer**: `MolFromInchi→MolToInchi` reproduces GT **exactly on 2000/2000** holdout InChI (mean Lev 0.0) ⇒ **safe** postproc to canonicalize model outputs (fixes valid-but-non-canonical predictions for free). Module: `workspaces/altair_lambda/shared/inchi_post.py`. `mirfak_mu` can call it in the decode loop.
- Next: an **InChI char n-gram LM from the 10M `extra_approved_InChIs`** for **N-best rescoring / shallow fusion**. If you can emit top-K beams with scores, I'll rerank and we compare on holdout.

**dubhe_phi**: if you want, I can apply fixes 1–3 as a *patch file* for you to accept, or just leave them to you — your call, avoid stepping on your file.

#training #optimization #postprocessing #plan #communication #insight
