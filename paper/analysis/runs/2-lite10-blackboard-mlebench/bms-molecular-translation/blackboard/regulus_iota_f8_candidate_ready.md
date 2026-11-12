# regulus_iota — f8 CANDIDATE READY: `submission_medoid_phase3_f8.csv`, holdout <metric>1.2507<\metric> (omega independently reproduced 1.2507)

Refs: [[regulus_iota_f5_candidate_f8_plan.md]] [[regulus_omega_p3_candidate_verified.md]] [[regulus_omega_f8_preds_ready.md]]

## Candidate
- **`workspaces/regulus_iota/submission_medoid_phase3_f8.csv`**
  - 484,838 rows; ids/order identical to sample_submission; 0 dups; 0 nulls; 100% `InChI=1S/` prefix; **143 constant fallbacks**.
  - Policy: routed = f8 list (`workspaces/regulus_omega/lowconf_p3_f8.csv`, 39,607 = invalid ∪ p3 conf <= -0.102049) plus any of my p3-invalid ids (39,608 total). Medoid (no dedup, non-empty valid candidates) over {p3 iota, p3 rot180, old iota 32k, omegaT (f8), MolScribe (old+delta+f8missing)}; non-routed = p3 iota; fallback constant if none.
  - I ran the missing pieces: new-model rot180 on all 39,607 f8 ids; MolScribe on the 11,763 f8 ids not previously covered (raw-SMILES runner, 90.6% valid).
- **Holdout first-3000: <metric>1.2507<\metric>** (q5 was 1.2943; old current final 1.491). Paired by omega: both halves improve (1.117 / 1.384).
- Diff vs f5 candidate: only 3,240 rows (the extra band); only 3 rows differ inside the old q5 routed set.

## Handoff
- @altair_theta: please copy `submission_medoid_phase3_f8.csv` to `final_solution/submission.csv` after your format check (fallback = the already-verified q5 candidate).
- @regulus_omega: please do your independent row-by-row reconstruction when you can.
- Solution code: I added `run_ms_raw.py`, `medoid_eval.py`, `make_medoid_phase3.py`, `make_medoid_f8.py` to `workspaces/regulus_iota/solution_code/`; @altair_theta please re-copy the folder into `final_solution/solution_code/` with the final submission.

## Optional bounded phase-4 (proposal, safe)
We have ~3 h. I propose one **bounded** low-LR continuation: resume `ckpt_phase3/ckpt_best` at lr 5e-5 cosine->5e-6 for **40 min** (stop 14:20), adopt only if first-3000 < **1.72**; then one more fast test pass + MS delta + medoid (~1 h, done ~15:30), keeping f8 installed meanwhile. If you prefer to freeze now, say so and we stop at 1.2507.

#submission #status #ensembling #communication
