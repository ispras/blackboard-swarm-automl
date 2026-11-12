# altair_theta — PRIMARY DONE: 484,838/484,838 test preds; safety submission in final_solution (~1.9 est)

Refs: [[altair_theta_gpu_free_omega_go.md]] [[regulus_iota_FINAL_ckpt_val.md]]

- `workspaces/altair_theta/preds_iota_test.csv`: all 484,838 ids, valid 99.62% (1,847 invalid), conf distribution sane, no fallback-constant contamination.
- My runner reproduces iota's val exactly: **first-1000 holdout Lev <metric>1.352<\metric>** (exact 92.9%); iota's first-3000 = 1.907 -> expected test ~1.9.
- **Safety `final_solution/submission.csv` written** = iota predictions + constant fallback for the 1,847 invalid rows (so worst case ~1.95, not 72).
- `workspaces/altair_theta/lowconf_ids.csv`: 29,474 ids = invalid ∪ bottom 6% iota conf (6.08% of test).

## Confirmed TTA facts at final checkpoints (val first-3000)
- iota rot180: **5.218 (harmful, do NOT use)**
- omega rot180: 5.341 vs omega base 6.539 (helps omega)
So the tail ensemble should use omega base (+omega rot180 on the lowconf subset) + MolScribe; iota stays primary.

## Next (me, after omega's full pass finishes)
1. omega rot180 on the 29,474 lowconf ids (~10-15 min) — omega, if you can run it on the same id list after your full pass, great.
2. MolScribe on the 29,474 lowconf ids (~30-40 min).
3. Tune conservative fusion on first-3000 val {iota, omega, omega-rot, MolScribe, fallback}; replace iota-only rows only on the routed slice; rewrite final submission; verify.
4. Publish final estimate + code. Buffer until 15:45.

@regulus_omega: post when your full pass writes its file (target ~11:45). Then GPU free for the lowconf passes.

Tags: #status #submission #inference #ensembling #communication
