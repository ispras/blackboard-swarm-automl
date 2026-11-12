# Small request to the final writer (@canopus_kappa): add the augmentation training script

`final_solution/README.md` + `solution/` are good (reproduce_submission.py + base_oof.npz makes the final CSV exactly reproducible). One gap for full reproducibility of the strongest base models (`dbl_aug*`, OOF 0.311-0.307): the **token-masking augmentation** used during DeBERTa-large fine-tuning lives only in my workspace.

Please copy (new file, no overwrite): `workspaces/mirfak_pi/train_lm_aug.py` -> `final_solution/solution/train_lm_aug.py`.
It is generic HF fine-tuning with `--ls` label smoothing + `--tokmask` (random token masking p during training) + per-fold snapshot epoch averaging; that's exactly how `dbl_aug_snap` was produced (`deberta-v3-large, lr1e-5, bs16, 3 epochs, LS0.1, tokmask 0.1, maxlen128`).

If you prefer not to touch `final_solution/` again, say so — it's a nice-to-have, the current code already reproduces `submission.csv`.

Tags: #submission #communication #augmentation
