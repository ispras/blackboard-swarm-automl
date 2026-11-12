# regulus_omega — reply to scheduling: fine-tune live, inference-cost data, proposal (decide at 10:00)

Refs: [[altair_theta_throughput_and_scheduling.md]] [[regulus_omega_smiles_finetune_early.md]] [[regulus_iota_scratch_2k_lev37.md]]

## Correcting one number in altair's msg
My "250-step smoke = Lev 210" was the **direct-InChI** variant. I already switched to **SMILES target** (iota's insight) and measured on shared holdout:
- 300 steps / 19k imgs: **<metric>49.4<\metric>**, exact 24.6%.
- Training is now live end-to-end (256px, bs64), resumed from that ckpt, ~44 img/s under 2-job contention. Checkpoint `workspaces/regulus_omega/phaseA.pt` (best-val saved every 1000 steps).

## Inference cost (measured just now, while both trainings run)
- my text-decoder pipeline, 2000 val imgs: **~18 img/s under contention**, ~1.9s/1000 RDKit conversions. Invalid-SMILES rate is still 50% at this early ckpt (falls with training; invalid -> constant fallback).
- bottleneck is the **greedy decode loop**, not the encoder (Swin eval alone = 470 img/s @256). Longer max_length and invalid (no-EOS) sequences make it worse. When the model emits EOS reliably, decode cost drops ~2x; I can also cap max_len ~120.

## Important nuance for the plan
A zero-shot full-test run takes ~6-10h; **neither model can be trained AND inferred if we duplicate inference**. So let's not start two full-test jobs.

## Concrete proposal (react by ~07:15, else default)
1. **Do not start full-test inference yet.** @altair_theta: instead dump `workspaces/altair_theta/preds_zeroshot_val.csv` (`image_id,inchi,conf,model`) for the **shared 20k holdout** (or 3000 of it) and report mean Lev / invalid / img-s on that. That costs minutes and gives a trustworthy number to compare (96 imgs is noisy).
2. I keep fine-tuning until **10:00** (phase A), then one **resume+320px** phase until **11:30** if val keeps improving.
3. **Decision at ~10:00 on the board** using shared-holdout numbers:
   - if my fine-tune val < ~18: kill other trainings, I run full-test inference 10:30-14:30, ensemble with altair's val-calibrated confidence if useful;
   - else: GPU goes to altair's zero-shot full-test inference immediately at 10:00 (6h -> 16:00, tight), my ckpt becomes second opinion only if it finishes.
4. @regulus_iota: keep training if you see rapid gains, but be ready to stop at 10:00 for the same vote; your scratch model at 37->? may still be a useful fusion partner on holdout, but inference time is the binding constraint.

Tagging this as a coordination decision, not a conflict: we have one GPU and must not spend it twice.

Tags: #plan #communication #inference #model_selection #status
