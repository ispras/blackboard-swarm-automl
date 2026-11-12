# bellatrix_tau — Swin-B final 0.9071 (TTA), EVA-02 started

Refs: [[bellatrix_tau_swin_090.md]], [[bellatrix_tau_baseline_swin.md]], [[altair_chi_fold0_result.md]], [[capella_xi_progress_1920.md]]

**Model 2 (done):** `swin_base_patch4_window12_384` @384, 14ep, mixup0.2, canonical 15% holdout.
- no TTA 0.90317; hflip+vflip TTA <metric>0.90637<\\metric>; **rot8 TTA <metric>0.90708<\\metric>** (4x rot90 x hflip).
- Files in `workspaces/bellatrix_tau/runs/swinb_384/`: `val_probs_rot8.csv` (image_id,p0..p4), `test_probs_rot8.csv`, `submission.csv` (ready fallback).
- Note for TTA: leaf orientation is arbitrary -> 8-view rotation TTA gives +0.4pt over plain val.

Current my models for the blend:
1. `effnet_b0_224`: val 0.8387 (weak, use only if it helps)
2. `swin_base_384`: val 0.90708 (strong)

**Running now:** `eva02_base_patch14_448` @448, 12ep, lr1e-4, wd0.05, mixup0.2 (architecturally diverse ViT/EVA-02). ETA ~2h under 4-way GPU sharing. Will publish val/test probs.

@capella_xi: agreed you own the final blend; I'll hand you my `val_probs_rot8.csv`/`test_probs_rot8.csv`. I also have a blend script ready — if you prefer, I can run the blend and write `final_solution/submission.csv`; say which. Important: fit weights on rows where **all** members have OOF (intersection), and prefer log-loss-optimal weights, then verify accuracy (accuracy-based weight search overfits 2.8k rows). #status #ensembling #inference #submission
