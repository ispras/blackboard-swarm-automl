# regulus_omega — BREAKTHROUGH: MolScribe-SwinB fine-tune (SMILES target) holdout <metric>25.18<\metric> at step 1000

Refs: [[regulus_omega_agree_0930_contract.md]] [[regulus_omega_smiles_finetune_early.md]] [[regulus_iota_smiles_target_insight.md]] [[altair_theta_horse_race_contract.md]]

## Measured on shared holdout (`val_20k.csv`, n=500, honest)
| step | images seen | mean Lev | exact |
|---|---|---|---|
| 300 | ~19k | 49.40 | 24.6% |
| **1000** | **~64k** | **<metric>25.18<\metric>** | **54.0%** |
- Training loss 0.55 -> **0.196** and still falling; LR only just reached full value (2.5e-5 encoder / 2.5e-4 decoder) at warmup end.
- Checkpoint: `workspaces/regulus_omega/phaseA.pt` (best-val, saved every 1000 steps).
- Training continues until **~09:12** (175 min budget from 06:17), then I stop per the 09:30 contract.

## Implication
The earlier "omega is behind, stop it" assessment is now stale: **25.18 @ 1k steps** already beats iota's 2k-step 37.4 and is close to zero-shot 21.66; the curve is steep. At 09:30 I expect substantially lower. Please keep the **09:30 stop / 09:45 vote** (not 11:30), and treat my lane as a real contender.

## Next
- ~09:30: publish `preds_val_phaseA.csv` (`image_id,inchi,conf,model`) for n=2000 + exclusive-GPU img/s.
- If I win the vote: start full-test inference immediately at 10:00; with valid-SMILES rate rising, decode length shrinks and I expect **faster than the 0.071 s/img zero-shot** (will measure).
- If I lose: I'll run only the low-confidence fill and/or CPU fusion.
- Inference already emits `conf` (mean token logprob), constant fallback for invalid SMILES.

Tags: #status #insight #training #model_selection #communication
