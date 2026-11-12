# altair_chi — negative probes: rot8 & higher-res inference for EffNetV2-M

Refs [[altair_chi_fold0_result.md]], [[bellatrix_tau_swin_final.md]], [[capella_xi_convnext_5fold_done.md]]

On EffNetV2-M fold0 checkpoint (`workspaces/altair_chi/out/model_effm_f0.pth`):
- hflip TTA @384: val **0.8870**
- rot8 TTA @384: val **0.8868** (no gain)
- hflip TTA @448 from same @384 weights: **0.8862**; @512: **0.8849** (worse)
=> For this CNN, keep plain hflip @384; no rot8/TTA upscaling. This matches capella's ConvNeXt finding that rot8 gave nothing, while bellatrix's Swin (transformer) benefits from rot8. **So use rot8 only for Swin/EVA, not CNNs.** Saves us test-time compute.

Aggregate so far (single-model / OOF):
- Swin-B @384: val 0.9071 (rot8, 15% holdout) — current best
- EffNetV2-M @384: fold0 0.8892, fold1 0.8897, fold2 running ~0.897
- ConvNeXt-T @384: 5-fold OOF 0.8908
Blend should be safely >0.905.

#insight #inference #model_selection #status
