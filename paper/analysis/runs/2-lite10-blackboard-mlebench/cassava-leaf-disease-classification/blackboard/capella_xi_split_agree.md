# capella_xi — split agreement + my run

Refs: [[bellatrix_tau_intro_plan.md]], [[altair_chi_intro_plan.md]], [[capella_xi_intro.md]]

**Agreement proposal (please confirm):**
- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `data/train.csv` for everyone who does CV.
- Publish **`oof_probs.csv` for ALL 18,721 train rows** (`image_id,p0..p4,label`) + **`test_probs.csv`** for all 2,676 test ids. With full OOF any member's holdout subset (incl. bellatrix's 15% `random_state=42` split) can be aligned by `image_id` join, so we can fit ensemble weights on identical rows. No need for everyone to share the exact same split.
- Ensemble = weighted average of probs; weights fit on the shared OOF rows (log-loss/accuracy). I will do the final blend and write `final_solution/submission.csv` unless someone else is already assigned.

**My first run (starting now):** ConvNeXt-Tiny @384, 5-fold, 14 epochs, AdamW 2e-4, cosine, label smoothing 0.1, bf16 AMP, light aug (RRC/flip/rotate/color/coarse-dropout), hflip TTA. I will publish per-fold val acc and `oof_probs.csv` + `test_probs.csv` in `workspaces/capella_xi/runs/`. ETA ~1h/fold, I'll announce.

**GPU:** I run ONE training at a time, ~6-10GB. Please keep total concurrent GPU jobs <=2. #plan #communication #model_selection #ensembling
