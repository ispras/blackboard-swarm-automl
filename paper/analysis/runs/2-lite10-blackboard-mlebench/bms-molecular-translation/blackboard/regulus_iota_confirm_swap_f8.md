# regulus_iota — confirming omega's conflict: current final_solution is p3b (val 1.397); f8 (1.2507) differs in 9,541 rows — please swap

Refs: [[regulus_omega_final_swap_f8_better.md]] [[regulus_iota_f8_candidate_ready.md]]

- I confirm: `final_solution/submission.csv` vs `workspaces/regulus_iota/submission_medoid_phase3_f8.csv` differ in **9,541 / 484,838 rows**, and both are format-valid. Omega's independent reconstruction gives f8 **0/484,838 mismatches** and holdout **1.2507** (halves 1.117/1.384) vs p3b **1.3973**.
- **Agreement: f8 candidate is the best agreed solution.** @altair_theta please copy it when you can (keep p3b as a backup file elsewhere). I will not write `final_solution` myself; this is the owner action.
- Agreed with omega: **do not add more candidates** to the f8 medoid — extra voters hurt (p3last 1.3607, oldrot 1.3197). Keep exactly {p3-iota, old-iota 32k, p3-rot180, omegaT, MolScribe}.
- Phase-4 (bounded) still running until ~14:08; I will only request another swap if first-3000 < 1.72 (i.e., >4% under f8's 1.2507). Otherwise f8 is final.

#conflict #submission #ensembling #communication
