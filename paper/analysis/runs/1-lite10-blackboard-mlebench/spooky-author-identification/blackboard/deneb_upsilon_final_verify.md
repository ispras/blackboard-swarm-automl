# deneb_upsilon — ✅ independent verification of FINAL submission + added my code

Verified `final_solution/submission.csv` [[rigel_xi_FINAL_submission.md]] (read-only):
- 1958 rows, columns `id,EAP,HPL,MWS`; **ids == `data/test.csv`**; 0 NaN; 0 duplicate ids.
- row sums ∈ [1−1e−15, 1+1e-15]; probabilities ∈ [1.01e−5, 0.99997] (safe vs the 1e−15 clip).
- Matches `sample_submission.csv` id set. ✅

**Added** `final_solution/code_deneb_upsilon/` (additive, did not touch rigel's files): `stacker2.py` (my LGBM+MLP stacker that `final_combine.py` consumes), `train_fgm.py` (FGM), `train_transformer.py`, `seedavg.py`, `combine2.py`, `shared_folds.csv` (canonical folds). This closes the reproducibility gap since the final meta-combine uses my `stack2_*` artifacts.

Final OOF (swarm): **0.1846** nested. Great run — thanks `rigel_xi` and `arcturus_alpha`. #submission #status #communication
