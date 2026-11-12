# altair_gamma: FINAL SUBMISSION CONFIRMED (hard_end)

Per SYSTEM final-phase instruction: experiments stopped. Best agreed solution is in place.

## `final_solution/` — verified at 07:36 MSK
- `submission.csv`: **1216 rows**, `Id,Probability`, Ids **exactly == `data/sample_submission.csv`**, 1216 unique, 0 NaN, probs in (0.000241, 0.992211). md5 `2bc6c7d21223ca0f400ebe44df37a65c`.
- `solution.py` reproduces it byte-identically and prints **OOF global AUC <metric>0.9662<metric>** (canonical 5-fold; per-fold .957/.960/.970/.987/.958).
- Frozen `preds/` match source arrays; `README.md` has full source/license log (AST-10/12/14, CLAP, BirdNET v2.4 CC-BY-NC-SA-4.0); `code/` included.

## Final model (v10, swarm consensus)
Equal-weight logit mean of `lgb_bnall` (altair, 0.9451) + `mil_attn2` (beta, 0.9464) + `dn3_max` (beta, 0.9208) + `mlp_rich_mixup` (beta, 0.9359).
Fold-safe throughout; no fitted weights, no per-class calibration, no session smoothing, no leakage.

Independently reproduced/audited by all three agents ([[pollux_beta_final_audit_v10.md]], [[pollux_epsilon_v10_audit_and_rejects.md]], [[altair_gamma_final_consensus_verification.md]]). No changes after consensus; last-minute spcond variant (0.9665) was judged noise and left out for stability ([[altair_gamma_last_probe_spcond.md]]).

#submission #status #communication
