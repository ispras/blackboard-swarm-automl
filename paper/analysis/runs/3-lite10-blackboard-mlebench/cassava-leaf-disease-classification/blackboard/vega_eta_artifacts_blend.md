# vega_eta — artifacts ready for blend (EVA02 honest+FT, CaFormer honest; CaFormer FT running)

@bellatrix_delta — paths for your blender (all test probs = sample_submission row order, hflip TTA; val probs = fold0 order, hflip TTA):

**EVA02-B@448** (`eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`)
- honest val: `workspaces/vega_eta/runs/eva02b448_f0_s42/val_probs_tta.npy` → fold0 acc **<metric>0.88865</metric>**
- honest test: `workspaces/vega_eta/runs/eva02b448_f0_s42/test_probs.npy`
- FT-all test: `workspaces/vega_eta/runs/eva02b448_f0_s42/full/test_probs.npy` ✅
- (no-TTA val was 0.88945; use TTA val for consistency with TTA test)

**CaFormer-b36@384** (`caformer_b36.sail_in22k_ft_in1k_384`)
- honest val: `workspaces/vega_eta/runs/caformer384_f0_s42/val_probs_tta.npy` → fold0 acc **<metric>0.88892</metric>**
- honest test: `workspaces/vega_eta/runs/caformer384_f0_s42/test_probs.npy`
- FT-all test: `runs/caformer384_f0_s42/full/test_probs.npy` (running, ETA ~02:00)
- (no-TTA val was 0.88732)

So both my models sit ~0.889 fold0 (TTA). Agreed with your finding: **uniform weights** — my independent blend check matches equal ≈ best. I'll run an independent verification of your final blend file when you post it (accuracy-agnostic: check prob ordering/labels/row count + that hedged vs honest-only label flips are small).

**Next (GPU free ~02:00):** I'll add one more *distinct* family for the ensemble — leaning `maxvit_base_tf_512.in21k_ft_in1k` @512 (multi-axis attention, high-res) or `convnextv2_base`@384, 10–12 ep, same fold0 split + FT. Will announce before launch so we don't collide with your B5@640 FT.

Refs: [[bellatrix_delta_blend_cv_insight.md]] [[vega_eta_phase2_ack.md]] [[vega_eta_phase3_ack.md]]

#ensembling #status #inference #feature_engineering

---
## Insight: per-class logit-bias / prior correction does NOT help (nested CV)
Ran `workspaces/vega_eta/postproc.py` on 6 honest models (EVA02, CaFormer, B5@512, B5@640, ConvNeXt, Swin). Uniform full-fold0 = **0.90067**.
Nested 5-fold CV inside fold0: uniform **0.90067**, per-class additive-logit-bias tuning **0.89533** (mean gain **−0.005**). So the minority-class weakness (CBB/CBSD/CGM/Healthy) is *not* fixable by class-prior/bias post-processing — it's genuinely model error. **Recommendation: keep plain uniform blend, no post-hoc class weighting.** Add diversity instead. (Per-class breakdown from [[aldebaran_epsilon_blend_check5.md]].)
