# dubhe_zeta — optional 4th diverse model (FPN/b4/320) training; include only if it helps holdout

Read [[dubhe_zeta_alldata_probs_ready.md]], [[mimosa_sigma_fold0_done.md]], [[betelgeuse_zeta_fold0_probs_ready.md]]. Since we have spare time and GPU is freeing up, I'm training **one extra diversity model** to potentially lift the blend:

- `segmentation_models_pytorch FPN + efficientnet-b4 (ImageNet), 2.5D 3ch, 320px, BCE+Dice, cosine warmup, 14 ep`, fold0.
- Different **decoder** (FPN) from the existing Unet(b4/5ch), Unet(b3/3ch), Unet++(b3/3ch) → useful diversity for averaging.
- Will publish by ~**04:30 MSK**: `workspaces/dubhe_zeta/handoff/val0_probs_fpn.npy` (fold0 order) and `workspaces/dubhe_zeta/handoff/test_probs_fpn.npy` (`cache/test_ids.json` order), plus its OOF combo from `eval_model.py`.
- **Rule: include only if it improves the honest holdout.** @mimosa_sigma you can just add it as a 4th candidate in the weight search; if its weight goes to 0 on the tune-half, drop it. If it arrives after your freeze, use the 3-model blend — no risk.

Current status: 3-model blend (b4v2 + my b3-Unet + betel's Unet++) is being tuned by mimosa; my fold0 OOF 0.9466, betel fold0 dice_all 0.9318, mimosa b4v2 dice_all 0.9274 — complementary. `final_solution/submission.csv` holds my validated fallback until mimosa's ensemble is published. #status #ensembling #architecture #communication
