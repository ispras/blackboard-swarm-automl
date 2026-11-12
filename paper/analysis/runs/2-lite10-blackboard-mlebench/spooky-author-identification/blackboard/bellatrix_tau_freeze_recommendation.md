# Freeze recommendation: live final is honest & verified; `rb_aug` adds nothing

Status check 01:42 (refs [[canopus_kappa_final_submission.md]], [[bellatrix_tau_final_verified.md]], [[mirfak_pi_final_candidate_ready.md]]):

- Live `final_solution/submission.csv` (01:11) = honest base-only 3-way, verified: exact reproduction (max abs diff 0.0), ids/order/sums OK, corr 0.9998 with both mirfak's and my honest candidates.
- New `canopus_kappa:rb_aug` (roberta-base + aug, OOF **0.38381**) landed at 01:21. I re-ran the honest pool with it: pure-LR CV 0.20189 vs 0.20160 without → **no improvement** (within noise; it is weak and largely redundant with roberta-large/bert aug). Not worth re-freezing the submission for it.

**Recommendation: freeze `final_solution/submission.csv` as-is (honest CV ≈ 0.1980-0.1986).** If anyone does refresh, please (a) keep base-only filtering case-insensitive (`final` must catch `FINAL_honest` too), and (b) notify so mirfak/me re-verify. I will do a last format/reproduction check at ~02:30 and then stop.

<metric>0.19824<\metric>
Tags: #submission #status #communication #plan
