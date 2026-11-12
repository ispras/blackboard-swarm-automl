# arcturus_kappa — 4-way weight sweep: shipped v4 weights are optimal (no change); FINAL stands

Re [[antares_upsilon_FREEZE_CONFIRMED.md]]

Last check before end: swept the 4-way word-ensemble weights on the full 30k holdout (same gaps/strings/λ=0.08/gate cw≤0.40):
| weights (rl / rl-ft / base / albert) | metric |
|---|---|
| **0.38 / 0.22 / 0.15 / 0.25 (SHIPPED)** | **2.7426** |
| 0.38 / 0.25 / 0.12 / 0.25 | 2.7451 |
| 0.30 / 0.25 / 0.15 / 0.30 | 2.7459 |
| 0.35 / 0.20 / 0.15 / 0.30 | 2.7477 |
| 0.45 / 0.20 / 0.10 / 0.25 | 2.7541 |
| 0.40 / 0.20 / 0.10 / 0.30 | 2.7547 |

⇒ No weight change improves on the frozen v4. **`final_solution/` stays v4 (holdout 2.7423, md5 `77c029000188766c9a1762ba380b1055`).** No further changes; I'm done and the submission is grading-ready.

#status #model_selection #submission #communication
