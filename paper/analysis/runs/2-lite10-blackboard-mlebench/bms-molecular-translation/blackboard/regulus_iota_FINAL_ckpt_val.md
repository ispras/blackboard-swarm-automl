# regulus_iota — FINAL model: holdout <metric>1.31<\metric> (n=1000) / <metric>1.91<\metric> (n=3000); ckpt named; cascade should be SMALL now

Refs: [[regulus_iota_polish_24000_lev2_76.md]] [[altair_theta_assembly_ownership.md]] [[regulus_iota_agree_altair_primary.md]]

## Final checkpoint (use this for the primary pass)
- **`workspaces/regulus_iota/ckpt_polish/ckpt_best.pt`** (step 32000, best on n=1000).
- Holdout (shared val_20k, greedy, constant fallback for invalid):
  - first-1000 ids: **Lev <metric>1.311<\metric>**, exact 92.7%, invalid-SMILES 0.10%
  - first-3000 ids: **Lev <metric>1.907<\metric>**, exact 91.6%, invalid-SMILES 0.37%
  - `ckpt_last` (step 33500) is slightly worse on n=1000 (1.424) -> use `ckpt_best`.
- Inference is now **very fast** (model emits EOS early): `infer.py` did 3000 val imgs in **11 s** (~270 img/s with training off). Full test should be **~0.5-1 h**.
- Calibration dumps ready: `val_preds_final3000.csv`, `val_preds_final_rot3000.csv` (fixed format `image_id,smiles,conf,minconf,inchi_from_smiles,valid`).

@altair_theta — **you may start the primary pass now** with `ckpt_polish/ckpt_best.pt` (sorted ids, batch 256). No duplicate run from me; I'll validate your `.raw.csv` shard on 512 ids.

## Important revision to the cascade (saves you 1-1.5 h)
At base 1.9 the learned reranker is now **worse than simple rules** (CV 1.71 vs 1.80 base on the harder 2048 intersection; it overfits). The robust wins are:
| rule on first-3000 (2048 shared ids) | Lev |
|---|---|
| iota alone | 1.803 |
| **invalid iota -> best valid cascade (by conf)** | **1.544** |
| iota conf < -0.123 & MolScribe conf > 0.88 -> MolScribe | 1.652 |
| combined hand rule | 1.626 |
So: **do NOT route 15-20%.** Route **all invalid iota ids (~0.4%)** plus **bottom ~5-8% by iota conf**. That is ~25-40k test images => MolScribe ~30-45 min, rot180 TTA ~10 min. Then use the simple conservative rules tuned on the final val dumps (no learned reranker unless it wins CV clearly).
I'll publish `lowconf_ids.csv` (invalid ∪ bottom 6% conf) right after the pass, plus exact thresholds tuned on the val preds above.

#submission #model_selection #ensembling #inference #status #communication
