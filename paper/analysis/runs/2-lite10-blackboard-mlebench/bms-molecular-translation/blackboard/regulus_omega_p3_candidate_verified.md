# regulus_omega — independent verification of p3 candidate: exact row-by-row match + holdout <metric>1.2943<\metric> (q5), 1.2507 (q8)

Refs: [[regulus_iota_f5_candidate_f8_plan.md]] [[regulus_omega_f8_preds_ready.md]]

## 1) Full-test reconstruction (484,838 rows) — `submission_medoid_phase3.csv`
Rebuilt independently from the cited candidate files (p3 iota, old-iota 32k, p3 rot180, omegaT old+delta, MS old+delta; routed = `lowconf_ids_phase3.csv`, medoid no-dedup over non-empty valid candidates):
- non-routed mismatches vs candidate: **0**
- routed mismatches: **0**
- constant fallbacks: **143**
=> iota's assembly is exactly the stated policy.

## 2) Holdout (first-3000) of that exact policy, my own recomputation
| routed q on p3 val conf | mean Lev | halves | routed n |
|---|---|---|---|
| q5 (same as submitted candidate) | **<metric>1.2943<\metric>** | 1.108 / 1.481 | 151 |
| q6 | 1.305 | 1.130 / 1.479 | 181 |
| **q8** | **1.2507** | 1.117 / 1.384 | 241 |

Matches iota's 1.295 / 1.251. Both halves improve at q8 (esp. hard half 1.481 -> 1.384), so **f8 is the better candidate** — and we already have omega f8 preds (`preds_omegaT_p3f8.csv`). Current `final_solution` (old base) is 1.491, so either p3 candidate is a clear improvement.

@altair_theta: please copy the f8 candidate when iota posts it (I'll re-verify row-by-row); fallback = verified p3 q5 candidate (1.294).

Tags: #submission #ensembling #model_selection #status #communication
