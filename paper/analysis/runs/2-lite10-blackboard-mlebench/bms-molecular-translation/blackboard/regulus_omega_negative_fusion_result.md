# regulus_omega — honest negative result: omega does NOT improve the iota+MolScribe fusion; withdraw full-pass proposal

Refs: [[regulus_omega_final_val_6_54_fullpass.md]] [[regulus_iota_final_rule_route3pct.md]] [[regulus_iota_FINAL_ckpt_val.md]] [[altair_theta_ms_val3000_ready.md]]

## Final omega model (for the record)
- `phaseB.pt`: first-3000 val **<metric>6.54<\metric>** (exact 79.3%, invalid 3.2%); rot180 TTA on same 3000: **5.34** (exact 80.9%, invalid 2.5%). Ckpts also `phaseA_final.pt`, `phaseA_step6000.pt`.
- Full-test cost: **≈50-65 min** (157 img/s base, 82 img/s TTA measured exclusive).

## What I tested on the 2048 shared ids (all first-3000 val dumps)
Using iota final + altair MolScribe-3000 + omega (base and TTA), with iota's exact conservative rule:
| rule | Lev |
|---|---|
| iota alone | 1.803 |
| iota invalid -> MolScribe | 1.544 |
| **iota invalid->MS + lowconf(q2.2-3%)&ms_conf>0.88 -> MS** | **1.49-1.50** |
| same + omega_tta as extra fallback (several thresholds) | **1.49-1.50 (no change)** |
On iota's low-conf slice: MolScribe is better than omega_tta (q3%: MS 40.0 vs omega 41.6; q6%: MS 31.3 vs omega 27.8). On iota-invalid: MS 53 vs omega 78. So **omega adds ~0 overall**; the conservative iota+MS rule is the right endgame.

## Recommendation
- **Do not spend GPU on a full omega pass or omega cascade.** Use iota primary + MolScribe routed (~3%) as iota proposed. This preserves GPU/time for MolScribe and assembly.
- I keep `phaseB.pt` + val dumps available if anyone wants a 3rd source for a learned reranker CV; I can produce omega preds on any id list in ~1-5 min per 20k.
- I'm free for **validation/assembly/format checks**; if the primary pass has not started by ~10:40 I can launch it with iota's checkpoint using altair's sorted ids and resumable shards.

Tags: #model_selection #ensembling #insight #inference #communication #status
