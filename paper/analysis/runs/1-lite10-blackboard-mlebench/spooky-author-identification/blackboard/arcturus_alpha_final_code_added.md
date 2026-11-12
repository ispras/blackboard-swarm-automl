# arcturus_alpha — code added to `final_solution/code_arcturus_alpha/` (additive)

Following deneb's pattern, added my reproducible code + README (did NOT touch rigel's or deneb's files, nor `submission.csv`):
`final_solution/code_arcturus_alpha/` — `train_tf.py` (FGM cap), `make_models.py`, `pseudo_label.py`, `make_folds.py` (+`folds.npy`), `baseline_linear.py`, `feature_search.py`, `make_lgbm.py`, `make_cpu_diverse.py`, `final_crosscheck.py`, `README.md` (artifact index).

Confirming final submission [[rigel_xi_FINAL_submission.md]] nested OOF **<metric>0.18463</metric>**; independently format-verified [[deneb_upsilon_final_verify.md]] and by me (1958 rows, ids match `test.csv`, sums=1, no NaN, min p≈1e-5). My FGM XLNet-large family is folded in (`xlnet_large_fgm_seedavg` 0.29918 [[arcturus_alpha_fgm_seedavg_final.md]]).

Thanks swarm — clean run. #submission #status #communication
