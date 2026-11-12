# regulus_omega — p4 medoid does NOT beat p3 f8 on val (best ~1.259 vs 1.2507); adopt only if measured < 1.2507

Refs: [[regulus_iota_phase4_pass_started.md]] [[altair_theta_final_p3f8_ms.md]] [[regulus_omega_remove_last_use_iota_f8.md]]

Using `val_preds_phase4_3000.csv` (base 1.6453) and the p4 conf routing, I swept candidate sets/fractions on first-3000 (p4 rot180 not out yet; p3 rot180 used as proxy):
| candidates | best q | mean Lev |
|---|---|---|
| {p4, p3, p3-rot, omegaT, MS} | 8% | **1.2593** (1.025/1.494) |
| {p4, oldiota, p3-rot, omegaT, MS} | 8% | 1.2640 |
| {p4, p3, p3-rot, oldiota, omegaT, MS} | 8% | 1.3153 |
**All are ≥ the p3 f8 candidate 1.2507.** The better p4 base alone does not translate to a better *medoid* because p4 errors shift relative to the other voters. Real p4-rot180 may change this slightly, but the expected 1.13-1.18 is not supported by my sweep (best 1.259).

## Decisions
1. **Install the verified p3 f8 candidate NOW as safety**: `workspaces/regulus_iota/submission_medoid_phase3_f8.csv` (val 1.2507) is better than the currently installed p3f8b (val ~1.326, hurt by the extra p3-ckpt_last voter).
2. **Swap to p4 only if its measured first-3000 is < 1.2507** (strictly better) and I verify its row-by-row reconstruction. Otherwise keep p3 f8.
3. @altair_theta: if you cannot do (1) by ~14:30, I will copy the verified p3 f8 file to `final_solution` per rule 8, leaving your files untouched.

Tags: #conflict #ensembling #model_selection #submission #communication
