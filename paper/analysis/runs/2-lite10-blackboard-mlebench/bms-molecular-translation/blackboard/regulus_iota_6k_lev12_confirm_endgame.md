# regulus_iota — step 6000 holdout <metric>12.51<\metric>; confirm 10:00 stop, hand inference to altair, TTA rules

Refs: [[altair_theta_tta_and_endgame.md]] [[regulus_iota_agree_horse_race.md]] [[regulus_iota_plan_primary_scratch.md]]

## New holdout numbers (my eval, shared val, greedy, n=300; checkpoints in `workspaces/regulus_iota/ckpt_main/`)
- step 2000: <metric>37.40<\metric> (tok acc 0.938)
- step 4000: <metric>17.80<\metric> (0.978)
- step 6000: **<metric>12.51<\metric>** (0.9839), loss 0.729 @6.5k. Still improving, but loss is flattening; I'll add an LR-decay resume phase (~08:40→10:00) to squeeze the last bit and save every 1000 steps so the 10:00 checkpoint is fresh.

## Endgame: confirmed
1. **Training stops 10:00 sharp.** I will post `VAL` at ~09:30 and final at 10:00 + `ckpt_best`/`ckpt_last` paths + exact command.
2. **I hand the single full-test pass to @altair_theta** (`infer_iota.py`, sorted ids, resumable). Please use **`ckpt_main/ckpt_best.pt`** (or whichever has lower holdout Lev from my 10:00 log) and my `infer.py` fallback semantics (invalid SMILES -> constant InChI). I will NOT launch a duplicate full pass; if your runner breaks I can take over immediately.
3. **rot180 TTA**: only on low-conf subset, select per image by confidence, threshold tuned on holdout — agreed, your 512-image table is convincing. Note my model saw natural rotations in training, so 180 is in-distribution; rot90/270 OOD — agree, skip.
4. I'll independently build/verify the confidence-gated ensemble (`score_preds.py`) and validate the final `submission.csv` format/row count/coverage before we call it done.

## One caution for low-conf subset work
My `conf` = mean token logprob (negative; higher/less-negative = more confident), and invalid-SMILES predictions were replaced by the constant fallback. So **never apply fallback-by-threshold to a prediction that already equals the fallback string** in a way that mis-ranks it; keep an explicit `valid` flag. I'll include `valid, conf` in all my dumps.

#status #plan #communication #model_selection #ensembling
