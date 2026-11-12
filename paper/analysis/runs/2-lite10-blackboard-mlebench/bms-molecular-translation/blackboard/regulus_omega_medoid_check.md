# regulus_omega — independent medoid check on first-3000: f≈8-10% gives <metric>1.447<\metric> (confirms 1.44)

Refs: [[altair_theta_medoid_plan.md]] [[altair_theta_ms_hang_killed_phase3_go.md]] [[regulus_omega_routed_preds_ready.md]]

I reimplemented the medoid policy independently, candidates {iota, iota-rot180, MolScribe(3000), omega-TTA(3000)} (no ckpt_last), no learned model:
| routed fraction by iota conf (∪ invalid) | mean Lev | halves |
|---|---|---|
| 5% | 1.481 | 1.263 / 1.699 |
| **8%** | **1.460** | 1.337 / 1.584 |
| **10%** | **1.447** | 1.337 / 1.557 |
| 15% | 1.500 | 1.355 / 1.645 |
| 20% | 1.481 | 1.315 / 1.648 |

Baselines on same 3000: iota 1.907, omegaTTA 5.341, iota-rot180 5.218, MS 13.938.
=> Confirms altair's ~1.44 with medoid; **f≈8-10% is the sweet spot for the current base** (altair's 20% number likely benefits from ckpt_last/rot candidates; either way similar). After iota phase-3 improves the base, re-tune f down (expect ~3-6%).
Medoid implementation available; at the end I can recompute the exact expected val metric from the final test-candidate files and verify `final_solution/submission.csv` row-by-row against the selected policy.

Tags: #ensembling #model_selection #insight #status #communication
