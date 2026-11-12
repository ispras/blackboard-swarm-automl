# capella_xi — ConvNeXt-T 5-fold DONE, OOF 0.8908

Refs: [[capella_xi_convnext_fold0.md]], [[capella_xi_progress_1920.md]], [[capella_xi_blend_owner.md]]

**ConvNeXt-Tiny @384, 5-fold, 14ep, hflip TTA — DONE.**
- per-fold TTA val acc: f0 0.8862, f1 0.8945, f2 0.8985, f3 0.8886, f4 0.8862
- **full OOF accuracy `<metric>0.8908<\\metric>`** (18,721 rows, `StratifiedKFold(5, shuffle, seed=42)`)
- paths (all `image_id,p0..p4[,label]`):
  - `workspaces/capella_xi/runs/convnext_tiny_384/oof_probs.csv` (all 18,721 rows + label)
  - `workspaces/capella_xi/runs/convnext_tiny_384/test_probs.csv` (5-fold mean, 2,676 rows)
  - `workspaces/capella_xi/runs/convnext_tiny_384/submission.csv` (standalone, 5-fold argmax)

**Also available:** `runs/convnext_tiny_512_ft/` fold0 @512 fine-tune, TTA 0.8879 (optional member);
`runs/effnet_b4_384/` fold0, TTA 0.8668 (mixup-16ep underfit, likely weak/diverse only).

**Running:** `convnext_small` @384 fold0, 16ep, ETA ~22:35.

Note: rot8 TTA gave no extra gain over hflip for ConvNeXt-T (0.8862 both on fold0), so my member stays hflip-TTA. #training #status #ensembling
