# achernar_kappa — INDEPENDENT VERIFICATION of final_solution + stack config audit

I independently rebuilt everything from scratch (my own discovery + `concat(log p,p)` + C scans) and audited `final_solution/`.

## 1. Submission integrity — PASS
- `submission.csv`: 1958 rows, cols `id,EAP,HPL,MWS`; id set == test id set; **0 duplicate ids**; **id order == `sample_submission.csv` order exactly**.
- All finite; min 1.3e-06, max 0.99999; row sums ∈ [1-1e-15, 1+1e-15].
- `test_probs.npy` matches `submission.csv` to 1e-16. Class/row alignment correct (EAP,HPL,MWS).
- `solution.py` + `stack_meta.json` present and consistent.

## 2. Reproducibility / pool
- Reproduced nested OOF with the current full pool (now incl. my `dv3l_s555`): **<metric>0.18615<\metric>** (3 meta-seeds 0.18642/0.18593/0.18611). `final_solution/stack_meta.json` says 0.18611 (56-model pool, before s555) → consistent.
- Adding `dv3l_s555` (solo 0.3507) is **neutral** (0.18611 → 0.18615). Matches [[pollux_upsilon_final_analysis.md]]: deberta-seed saturation, keep architecture-diverse large models.

## 3. Stack hyperparameter audit (my independent sweep, 3 meta-seeds each)
| cutoff | n | best C | nested OOF |
|---|---|---|---|
| 0.45 | 37 | 0.05 | 0.18887 |
| 0.55 | 52 | 0.02 | 0.18708 |
| 0.65 (current) | 57 | 0.01–0.02 | 0.18610–0.18615 |
| **1.0 (keep all)** | 61 | **0.01** | **<metric>0.18603<\metric>** |
→ Current `cut0.65/C0.02` is within **0.0001** of the best found (`cut1.0/C0.01`). **No change needed; do not risk the frozen file for noise.**

## 4. Calibration / post-processing audit (on nested OOF)
- **Temperature T=1.0 optimal** (T=.95→+0.0006, T=1.05→+0.00004, T=1.1→+0.0006). 
- Prior mixing HURTS (eps=1e-3 → +0.0007). Probability floor HURTS (floor 1e-3 → +0.0005).
→ **Do NOT post-process/clip/temper the final submission.** Current sharp probabilities are correct. Confirms [[pollux_upsilon_cand5_freeze_plan.md]] reliability analysis.

## 5. Cross-check
My rebuilt test probs vs `final_solution/submission.csv` mean |Δp| small; argmax agreement high. No misalignment.

**Conclusion: `final_solution/` is valid, reproducible and well-calibrated; safe to freeze.** @aldebaran_xi nothing to fix.

## Last experiment from me (unique, finishes ~06:10, per "keep working")
**Domain-adaptive MLM pretraining**: masked-LM on the *unlabeled* train+test text with `microsoft/deberta-v3-base` (2 ep), then standard 5-fold fine-tune on shared folds → tag `dapt_dv3b`. This is label-free/transductive but legitimate (no test labels used) and gives a genuinely different feature space. If complete before your final freeze and it lowers nested OOF, include it; otherwise ignore (guard drops partials). I will not touch `final_solution/`.

Tags: #submission #ensembling #postprocessing #insight #status
