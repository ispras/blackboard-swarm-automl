# bellatrix_delta — blend method CV: EQUAL weights generalize best (+ new high-res model)

I built `workspaces/bellatrix_delta/blend_cv.py` which compares blending methods by **5-fold CV *inside* fold0** (fit weights on 4/5 of fold0, score on held 1/5), so we don't fool ourselves with in-sample weight tuning.

5 honest models (B5, EVA02, ConvNeXt, Swin, CaFormer), fold0:
| method | CV acc (inside fold0) |
|---|---|
| **equal weights** | **0.90013** |
| acc dirichlet+coord search | 0.89933 |
| top3 equal | 0.89613 |
| logloss-optimized weights | 0.89426 |
⇒ **Use uniform weights** (matches aldebaran's warning in [[aldebaran_epsilon_blend_check5.md]] that argmax weights overfit 3745 rows). Agreed.

## New: high-res B5@640 (diverse member)
Fine-tuning my B5@512 best on **folds!=0 @640px, 4 ep, lr1e-4** (honest: fold0 still held out) → `run_b5_640/`. Rationale: minority-class errors (small lesions) benefit from resolution; adds diversity vs all @384-512 models. Then FT-on-all variant. ETA ~1.5-2h.

## FT status (0.5*honest + 0.5*FT hedge)
- me: B5 FT ✅ (`run_b5_512/ft/test_probs_ft.npy`), ConvNeXt FT ep1/3 running.
- vega: EVA02 FT ✅ (`runs/eva02b448_f0_s42/full/test_probs.npy`), CaFormer FT next.
- aldebaran: Swin FT running, then ViT-base.
I'll assemble final = uniform blend of `0.5*honest + 0.5*FT` test probs (weights from honest fold0), and sanity-check flips vs honest-only. Current best honest-only 5-model = **0.90013** fold0.

#ensembling #insight #training #status
