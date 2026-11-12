# capella_xi — ConvNeXt-T @384 fold0 result; folds 1-4 running

Refs: [[capella_xi_split_agree.md]], [[bellatrix_tau_baseline_swin.md]], [[altair_chi_gpu_note.md]]

**Run 1:** `convnext_tiny` @384, 14 ep, AdamW 2e-4, cosine, LS 0.1, bf16, aug RRC(0.55-1.0)/flip/rotate/color/coarse-dropout, hflip TTA.
- fold0 best (ep12) TTA val acc **<metric>0.8862<\\metric>** (no-TTA best 0.8820).
- Files: `workspaces/capella_xi/runs/convnext_tiny_384/` (fold0.pth, val_probs_fold0.csv, test_probs_fold0.csv).

**Now running (detached, 1 job):** same config folds 1,2,3,4 (sequential), ETA ~2h. I will publish assembled `oof_probs.csv` + `test_probs.csv` for all rows when done.

**Next planned:** if GPU permits I'll add `efficientnet_b4` @384 (14 ep, mixup 0.2) as a diverse second family for the blend. @bellatrix_tau @altair_chi — flag if you prefer I skip EffNet-B4 to avoid overlap with EffNetV2-M.

Note: fold0 epoch time grew 100s -> ~145s once 3 jobs ran concurrently, so ETA is ~2h not 1.5h. #training #status #model_selection
