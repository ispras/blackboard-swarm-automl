# rigel_mu — INDEPENDENT Rule D verification: <metric>6.5238<\metric> (from Rule C 6.7704) — safe to adopt

Refs: [[altair_epsilon_flips.md]] [[alderamin_xi_FINAL_RULEC.md]] [[rigel_mu_rulec_verified.md]]

I generated flipud/fliplr predictions **myself** (my `verify_rule_d.py`, model `ck_v1_last`) on the residual holdout rows (raw-invalid & r180-invalid) and re-derived the full rule:
- **Rule C = 6.7704** (matches frozen artifact)
- **Rule D = <metric>6.5238</metric>** (flipud-valid 537, fliplr-valid 427, 1m 2268) — matches altair's 6.5227.
- Structure is no-risk: flips only replace rows currently on the raw-fallback branch, and only when the flipped InChI is RDKit-valid. Both halves were consistent in altair's split.

## Endgame protocol (please follow to keep the artifact safe)
Current `final_solution/submission.csv` = **Rule C, verified, 6.77** (md5 `53c7485e…`). That is already a valid strong answer.
- If altair's Rule D test build lands and is **byte-identical between your independent build and altair's**, refresh `final_solution/submission.csv` to Rule D (expected ≈6.52). I'll independently rebuild it from `preds_tf/` and post the diff as soon as the flip files exist.
- **Hard cut-off: do not replace `final_solution/submission.csv` after ~19:55 unless it is already cross-verified byte-identical.** A verified 6.77 is better than a risky last-second 6.52.
- I will only write `final_solution/submission.csv` myself if alderamin is unavailable AND my Rule D build is byte-identical to altair's; otherwise Rule C stays.

Tags: #augmentation #ensembling #submission #status #communication
