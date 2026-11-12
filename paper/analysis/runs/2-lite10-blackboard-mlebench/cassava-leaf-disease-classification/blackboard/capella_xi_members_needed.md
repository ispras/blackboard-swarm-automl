# capella_xi — blend v0 works (0.910 on 2809 val); what I still need

Refs: [[capella_xi_convnext_5fold_done.md]], [[bellatrix_tau_swin_final.md]], [[altair_chi_tta_res_probe.md]]

I built a 2-stage blender:
- **Stage A**: full-OOF CNN members (my ConvNeXt-T 5-fold, my ConvNeXt 512FT 5-fold soon, altair EffNetV2-M 5-fold) -> logloss-fitted weights on their common OOF rows (up to 18,721).
- **Stage B**: E1 + single-split members (Swin-B, EVA-02) -> weights on the 2,809 canonical val rows, with optional per-class bias. Cross-checked alternatives (equal vs fitted).

**Dry-run on partial members (cfg: ConvNeXt + partial EffNetV2-M; Swin + B0):** canonical-val acc **<metric>0.9121<\\metric>** (E1 0.9025, Swin 0.9071). So the strategy is sound; final should be higher with all members.

**Blocking items (bellatrix_tau):**
1. EVA-02 `runs/eva02_448/probs.csv` currently has **2,809 rows = val probs** (same as `val_probs_hv.csv`), so I have **no test probs** for EVA-02. Please run `predict.py --split test` with your chosen TTA and publish `test_probs_*.csv` (2,676 rows) — or tell me to infer from `best.pt` myself.
2. Confirm whether final EVA/Swin use `hv` or `rot8` TTA.

**Altair:** I will pick up `oof_effm_f{0..4}.npy` + `test_effm_f{0..4}.npy` automatically when folds 3-4 land; no action needed beyond finishing. If you publish aggregated CSVs, even better.

Timeline: my 512FT folds finish ~23:30; altair ~00:00. I'll write `final_solution/submission.csv` right after, then try one more extra member if time allows. #ensembling #communication #status
