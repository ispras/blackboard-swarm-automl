# pollux_beta: FINAL SUBMISSION CONFIRMED (hard_end) — v10 in place

Per SYSTEM final-phase instruction ([[2026-10-31_07-21-55_SYSTEM_final_phase.md]]): experiments stopped, no further changes.

- `final_solution/submission.csv` md5 **`2bc6c7d21223ca0f400ebe44df37a65c`**, 1216 rows, `Id,Probability`, Ids exactly == `data/sample_submission.csv`, no NaN, probs in (0.000241, 0.992211).
- `python final_solution/solution.py` reproduces it byte-identically, prints OOF global AUC **<metric>0.9662<metric>** (per-fold .957/.960/.970/.987/.958).
- Model (swarm consensus v10): equal-weight logit mean of `lgb_bnall` (altair 0.9451) + `mil_attn2` (beta 0.9464) + `dn3_max` (beta 0.9208) + `mlp_rich_mixup` (beta 0.9359). Fold-safe; no test labels/leakage, no fitted weights, no session smoothing.
- `README.md` (source/license log: AST-10/12/14, CLAP, BirdNET v2.4 CC-BY-NC-SA-4.0), `preds/` (frozen, bit-identical to sources), `code/` included.
- Last-minute `v10 + spcond` (0.9665) judged within noise and left out for stability ([[altair_gamma_last_probe_spcond.md]]).

Confirmed by all three agents. Done. 🐦

#submission #status #communication
