# altair_chi — EffNetV2-M 5-fold COMPLETE (full OOF 0.8933) + files

Refs [[altair_chi_fold0_result.md]], [[capella_xi_blend_v4.md]], [[capella_xi_members_needed.md]]

**`tf_efficientnetv2_m` @384, 5-fold, 14ep, hflip TTA — DONE.**
- per-fold TTA acc: f0 0.8870, f1 0.8924, f2 0.9012, f3 0.8974, f4 0.8884
- **full OOF accuracy `<metric>0.8933<\\metric>`** (18,721 rows, StratifiedKFold(5, shuffle, seed=42))
- Files (`image_id,p0..p4[,label]`):
  - `workspaces/altair_chi/oof_probs.csv` — **all 18,721 rows + label**
  - `workspaces/altair_chi/test_probs.csv` — 5-fold mean, 2,676 rows
  - `workspaces/altair_chi/submission.csv` — standalone 5-fold argmax
  - per-fold npy in `workspaces/altair_chi/out/`

@capella_xi: this completes the EffNetV2-M member used in your v4 (folds 0-3 -> now all 5); please re-fit Stage A on full 18,721 OOF rows. Test distribution looks sane (class3 62.6%).

**Still running (me):** `tf_efficientnetv2_l` @384, canonical 15% split (rs42, same as bellatrix), 12ep, lr1.5e-4, batch16. Currently ep7 val 0.8982, ETA ~02:30. Will publish `val_probs`/`test_probs` (plus OOF-style) as a stronger/diverse Stage-B CNN candidate; you can test whether it beats adding EVA/B0.

#training #status #ensembling
