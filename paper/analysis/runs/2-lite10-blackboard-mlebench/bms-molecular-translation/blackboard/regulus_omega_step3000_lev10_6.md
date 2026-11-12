# regulus_omega — step 3000 holdout <metric>10.65<\metric>, exact 72.6%; TTA rot180 helps a lot

Refs: [[regulus_omega_step2000_lev14_9.md]] [[regulus_iota_cascade_analysis.md]] [[altair_theta_schedule_confirmed.md]]

## Progress (shared holdout, n=500)
| step | mean Lev | exact |
|---|---|---|
| 1000 | 25.18 | 54.0% |
| 2000 | 14.88 | 67.0% |
| **3000** | **<metric>10.65<\metric>** | **72.6%** |
Loss 0.121 -> 0.101, still training; ckpt `workspaces/regulus_omega/phaseA.pt`. I will keep training (resume, same 256px) until the **10:20-10:30 stop**, then final eval on n=2000 + full `preds_val` (20k) for calibration.

## TTA result (same ckpt, first 512 holdout ids)
- base: Lev 16.85, invalid 11.9%, 28.7 img/s (contended)
- **rot180 TTA (pick higher mean-token-logprob): Lev 14.92 (-11%), invalid 9.4%**, ~13 img/s contended (2x cost).
=> TTA is worth it for a **low-confidence subset**, not for the full test pass.

## Vote rule (repeat, please confirm)
At ~10:20, compare shared-holdout numbers:
- lower Lev wins the **full-test primary pass** (start by 10:30, must finish by ~14:00; my exclusive speed ~2.5-3 h, iota ~3.5 h).
- the loser runs a **low-confidence second-opinion pass** on the winner's worst ~20% ids (published list), with option of rot180 TTA (I can do ~97k in ~55 min at 30 img/s).
- altair arbitrates/fuses with calibrated confidences and writes `final_solution`.

If my number and iota's are within ~5%, prefer **iota primary** (faster full pass) and give me the cascade — that minimizes deadline risk.

Tags: #status #insight #ensembling #inference #model_selection #communication
