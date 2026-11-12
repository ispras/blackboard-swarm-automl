# vega_eta — ACK phase plan; my phase-2 pick = CaFormer-b36 (unique family)

Read [[bellatrix_delta_phase_plan.md]] — strong plan, I'm in.

**Phase 2 pick (final):** `caformer_b36.sail_in22k_ft_in1k_384` @384 (timm, MetaFormer hybrid, SAIL-22k→1k, HF). Rationale: max diversity vs EVA02 (ViT), Swin (aldebaran), EffNet-B5 + ConvNeXt (bellatrix); nobody has CaFormer. I benchmarked: 95.7M params, 0.345 s/bs16 fwd (vs convnextv2 0.221s) — acceptable. Weights pre-downloaded. 12 ep, AdamW lr 1.0e-4, cosine+warmup, same aug/label-smoothing + hflip TTA. Target start ~21:50 (right after my EVA02 ends, keeps ≤3 concurrent).

Note vs your table: I'll use CaFormer instead of EffNetV2-M — EffNetV2 would duplicate the EffNet-B5 family; CaFormer adds a genuinely new one. If you object, ping fast; otherwise CaFormer it is.

**Phase 1 status EVA02-B@448 (fold0):** ep4 **<metric>0.87931</metric>**, ep5 0.87557 (best 0.8793 so far, 6 ep left, lr decaying). Healthy. ETA ~21:45. [[vega_eta_family_locked.md]]

**Phase 3:** I agree with full-data fine-tune of the fold0 models (2–3 ep @ low lr) as a cheap "see-all-data" variant; suggest we *average* phase-2 (fold0-honest) test probs with phase-3 (full-data) test probs per model, reusing phase-2 blend weights — hedges any overfit. I'll help implement/verify. I'll also **independently sanity-check the blend** on fold0 before you write final_solution.

#communication #plan #model_selection #ensembling

---
## Phase 2 DONE — CaFormer-b36@384 final
12/12 ep, **best fold0 val_acc <metric>0.88732</metric>** (ep8; no-TTA eval). Trajectory .872 .881 .887 .852(dip) .884 .887 .887 .886 .887 .882 .883.
Artifacts `workspaces/vega_eta/runs/caformer384_f0_s42/`: `val_probs.npy` (fold0, no-TTA), `test_probs.npy` (hflip TTA, sample_submission order), `best.pth`, `submission.csv`.
→ Now **Phase 3**: EVA02 full-data FT (3ep, lr3e-5) running, then CaFormer FT. Computing hflip-TTA val probs for both now (first GPU lull, ~22+14+12GB used). [[bellatrix_delta_phase3_proposal.md]] [[vega_eta_phase3_ack.md]]
Note for blender: EVA02 <metric>0.88945</metric>, CaFormer <metric>0.88732</metric> (both fold0, no-TTA).
