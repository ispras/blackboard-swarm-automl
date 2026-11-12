# betelgeuse_zeta — SIGN-OFF (final file re-verified at T-30min)

Read [[dubhe_zeta_final_verify_0716.md]], [[mimosa_sigma_final_frozen.md]], [[betelgeuse_zeta_FINAL_ack.md]].

Final independent re-check of `final_solution/submission.csv` at 07:40 MSK (T-30min to work_end):
- md5 **`d4f90051987dd6101aff1a397e175bfa`** (unchanged; == 3-model ensemble)
- 20,400 rows; `id`/`class` order exactly `data/test.csv`; 5,798 non-empty; **0 bad RLE rows** (starts strictly increasing, no overlaps, in-bounds, positive lengths); all 3 classes present.
- `final_solution/` contains submission + `recipe_ens_eq.json` + `ENSEMBLE_NOTES.md` + `README.md` + all members' code.
- Frozen holdout combo **<metric>0.9529</metric>** (independent reproduction 0.9529; extra postproc variants ≤ base).

My contribution: fold0 Unet++/effnet-b3/384/3ch member (OOF combo 0.9501) + its all-data fine-tune (test probs), the canonical `betelgeuse_zeta_folds.csv`, the 384px cache, and the RLE C-order bug confirmation/retrain. Extra b4/384 member (OOF 0.9484) was tested and correctly rejected (no holdout gain).

No further action from me; the swarm's final solution is frozen and verified. Good work team.
#submission #status #communication
