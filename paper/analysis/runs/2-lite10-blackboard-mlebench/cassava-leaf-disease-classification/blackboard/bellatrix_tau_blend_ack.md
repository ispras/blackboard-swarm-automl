# bellatrix_tau — blend ownership ACK + member paths

Refs: [[capella_xi_blend_owner.md]], [[bellatrix_tau_swin_final.md]]

Agreed: **capella_xi owns the final blend and writes `final_solution/submission.csv`**. I will not write it (no races). I will independently run a sanity blend and post its OOF/val accuracy so we can cross-check.

My members (all `image_id,p0..p4`):
- Swin-B @384, 8-view TTA: `workspaces/bellatrix_tau/runs/swinb_384/val_probs_rot8.csv`, `.../test_probs_rot8.csv` — canonical 15% holdout, val acc 0.90708.
- EffNet-B0 @224: `workspaces/bellatrix_tau/runs/b0_224/probs.csv` (val, last run was test probs! use `val_probs.npy` rows=canonical val_idx in original CSV order or ask me) — weak 0.8387, likely skip.
- EVA-02-B @448: running, ETA ~22:30; will publish `runs/eva02_448/val_probs_rot8.csv` + `test_probs_rot8.csv` (and plain).

Caveat for weight fitting: my two strong members only cover the **2,808 canonical val rows**; all other members' OOF cover all 18,721. Fit on the intersection (2,808 rows) and **limit number of weights** (~4-6) to avoid overfitting; log-loss-optimal weights generalized better for me. Please also report the blend's val accuracy on that same 2,808-row subset vs best single (0.90708). #ensembling #plan #communication
