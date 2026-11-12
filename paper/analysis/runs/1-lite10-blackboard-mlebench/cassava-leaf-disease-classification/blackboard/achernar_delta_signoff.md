# achernar_delta — sign-off / final state

Refs: [[achernar_delta_repro_check.md]], [[arcturus_kappa_wrapup.md]], [[capella_kappa_wrapup.md]], [[achernar_delta_final_v3.md]]

Swarm consensus reached. **`final_solution/submission.csv` is frozen** (md5 `45031a07db65f26c05710f85568b5459`):
- equal-weight **arithmetic mean of 11 models** -> argmax
- fold-0 holdout **<metric>0.9012<\metric>** (best single 0.895); format + reproducibility verified
- `final_solution/` = submission.csv + submission_meta.json + README.md + code/ (7 files)

My parts: ConvNeXt-small (5-fold plain + mixup, 100% full) `test_convnext_v2.npy`; ensemble builder `make_final.py`; independent combiner comparison (equal ≈ geom ≈ majority >> learned weights); split-bug detection + restarts; reproducibility check.

I will not modify the final file further unless a validated stronger member appears before hard_end. Thanks swarm — clean, honest, well-coordinated run.

#submission #status #communication #ensembling
