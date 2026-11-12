# aldebaran_omicron — Model A_C FINAL holdout probs READY (C-space, 4× TTA)

Model A `runs/modelA_C_f0/best.pt` = b4 UNet, 2.5D (z-1,z,z+1), 320, C masks, stopped at ep9/16 (best val; plateau). **4× flip TTA** (identity+h+v+hv).

## Holdout (shared 15 cases, 6864 slices) — C-space
- per-slice dice **0.9250** (lb .9224 / sb .8998 / st .9528); positive-only [.832 / .701 / .827]
- 3D HD proxy hdn **0.0739** → HD score **0.926**
- **combo proxy ≈ <metric>0.926<\metric>** (0.4*dice + 0.6*(1-hdn), my HD normalization — trust canopus's full-val harness for absolute combo)

## Artifacts for fusion (join by `id`, C-space, uint8 prob*255, shape (N,3,320,320), channels lb/sb/st)
- val: `workspaces/aldebaran_omicron/probs/val_A_AC_probs.npy` + `val_A_AC_probs_index.csv` (6864)
- test: `workspaces/aldebaran_omicron/probs/test_A_AC_probs.npy` + `test_A_AC_probs_index.csv` (6800)
@canopus_phi — ready for [[canopus_phi_interim_fused_B6.md]] fusion now. Encode **C**; per-class thr prior lb .45 / sb .50 / st .50 ([[regulus_alpha_postproc_v2b.md]]); no CC cleanup on corrected models.

## Next
All-data fine-tune running now (`runs/A_C_alldata`, 6 ep, init from ep9, lr 1.5e-4, C cache) → ETA ~00:15; then `test_A_AC_alldata_probs.npy` for the final blend. The honest holdout fusion above remains the primary/fallback; the all-data variant should help seen test cases (19/28).

#status #inference #ensembling #metric #communication #submission
