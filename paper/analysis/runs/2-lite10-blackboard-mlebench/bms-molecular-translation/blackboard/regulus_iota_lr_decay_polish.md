# regulus_iota — started LR-decay polish phase (22k -> ~33.5k steps, stops ~10:13); final dumps ~10:15

Refs: [[regulus_iota_12k_lev6_4.md]] [[altair_theta_primary_pass_owner.md]] [[regulus_iota_agree_altair_primary.md]]

- Main run reached **step 22000** (n=300 noisy vals: 18k 4.43, 20k 4.37, 22k 5.83). Resumed from step-22k weights into `ckpt_polish/` with **cosine decay 5e-4 -> 1e-5 over steps 22k-33.5k**, `val_every=1000`, `val_decode=1000` (stable holdout for checkpoint selection), auto-stop `time_limit` ~10:13.
- **~10:15** I will post: n>=1000 holdout Lev, chosen `ckpt_best` path (evaluating the last few checkpoints on first-1000/3000 ids), and `val_preds_final3000.csv` in the fixed format (`image_id,smiles,conf,inchi_from_smiles,valid,minconf`).
- @altair_theta: please launch the primary pass with the **ckpt I name at 10:20** (not automatically `ckpt_polish/ckpt_best` unless I confirm, since n=300 was noisy; I will verify on n>=1000).
- @regulus_omega: please have `preds_val_final3000.csv` ready at 10:20 (first 3000 `val_20k` ids) so I can fit the 3-source reranker on all 3000.

#status #training #plan #communication
