# regulus_omega — phase-3 medoid tuning: route **f≈8%** (not 5%) if MS covers it -> val <metric>1.406<\metric> vs 1.436

Refs: [[regulus_iota_phase3_adopted.md]] [[altair_theta_phase3_adopted.md]] [[regulus_omega_medoid_check.md]]

Using the **phase-3** val preds (`val_preds_phase3_ckpt_best_3000.csv`, base 1.773) + phase3 ckpt_last + omegaT + MS + **old** rot180 (phase3 rot180 not yet available; proxy):
| routed f by iota conf (∪ invalid) | MS available only ≤5% | MS available to f |
|---|---|---|
| 5% | **1.436** | 1.436 |
| 6% | 1.447 | 1.443 |
| **8%** | 1.446 | **1.406** |
| 10% | 1.446 | 1.431 |
| 12% | 1.446 | 1.446 |

With {I3, phase3-last, omegaT, MS, old-rot180} and MS to f=8%: **1.406**, halves 1.231/1.580.
=> The extra 3% band is only worth routing **if MolScribe also covers it** (cost ≈ 14.5k extra ids ≈ 12 min at 20 img/s). OmegaT alone adds nothing on that band.
Caveat: n=3000, gain 0.03 is near noise (halves disagree: f5 1.159/1.713 vs f8 1.231/1.580); phase-3 rot180 may shift the optimum. Still, f=8% is the better point estimate and we have time.

## Proposed (owners decide)
1. When `preds_iota_p3.csv` is done (~12:46), compute routed list at **f=8%** (invalid ∪ conf ≤ val q8) — I can produce `workspaces/regulus_omega/lowconf_p3_f8.csv` immediately from the raw conf.
2. Altair runs robust MS on f8 (24.4k -> ~39k ids, ~25-30 min); I run omegaT on the **delta** (old routed -> f8) (~5 min) and reuse existing omegaT for overlap.
3. Medoid on f8, verify. Expected ~1.41 vs current 1.49 (old base) / ~1.44 (phase3 at f5).
If you prefer to keep f=5%, no problem — phase3 still improves the current submission.

Tags: #ensembling #model_selection #insight #plan #communication
