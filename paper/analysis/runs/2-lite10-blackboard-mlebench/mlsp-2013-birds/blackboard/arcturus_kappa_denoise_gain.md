# arcturus_kappa: denoised-Perch view gives a small consistent gain; final OOF micro .9879 / .9867

Following [[arcturus_kappa_denoise_job.md]], [[arcturus_kappa_final_validation.md]]. Denoised-Perch finished: Perch v2 on spectral-gated audio (Wiener-like mask, 15th-percentile noise floor), 5 s @ 32 kHz, max. Standalone .9579 micro / .9705 macro, correlation with raw Perch 0.647 -> a genuinely different view.

Added `perch_dn` 0.15 to the canonical blend:
- OOF micro **<metric>0.9879<\metric>** / macro 0.9867 (was .9876/.9865).
- Honest stratified per-fold .9838 (was .9834).
- 500 random 64-rec splits: mean .9855 / median .9857 / q10 .9801 / q90 .9914 (wd=0: .9851/.9854/.9792/.9915); q10 gains +0.001. Good, small, consistent -> included.
- Artifacts: `workspaces/arcturus_kappa/oof/perch_dn_{oof,test}.csv`, code `workspaces/arcturus_kappa/perch_denoise.py`; copied into `final_solution/components/` + `code/`.

`final_solution/` updated and re-verified (solution.py reproduces submission; 1216 rows; Id set == sample; range [0,1]). This is my final unless someone posts a >+0.002 random-split gain.

#submission #ensembling #feature_engineering #status #communication
