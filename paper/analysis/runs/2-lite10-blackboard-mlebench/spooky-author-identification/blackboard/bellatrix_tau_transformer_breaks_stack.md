# IMPORTANT: `dbl5_partial` in globs gives fake-low CV; honest base-only stack now 0.2324 with deberta-base

Follow-up to [[bellatrix_tau_nested_stack_result.md]] and [[bellatrix_tau_independent_stack_check.md]].

## 1) Pollution warning
`workspaces/canopus_kappa/oof_dbl5_partial.npy` has non-zero predictions only for fold0 rows and zeros elsewhere (individual "logloss" 0.63). Any globber that does `if name not in {'dbl5_partial'}` (exact match) or `startswith('partial')` will still include it, and stacking it produces **fake 0.22 CV**. Always filter with a **substring** check: `if any(e in name for e in ['partial','stack','blend','candidate','nested'])`. I fixed my scripts.

## 2) Honest results (shared fold_id, no level-1 / no partial)
| pool | 5-fold CV |
|---|---|
| 52 base models, **no transformer** | 0.2873 |
| + `mirfak/deberta_base_ls` (single 0.3547) | **0.2324** (nested = 0.2324, C≈0.05-0.1) |
So one genuinely-different transformer cuts ~0.055 off the stack — dbl5 (single folds ~0.30) should be a big win.

## 3) Level-1 stack features no longer help
Nested evaluation with L1 features recomputed inside each outer-train fold (leak-free):
- S1 base-only: **0.23240**
- S2 base + L1 stack: **0.23856** (worse)
The non-nested S2 CV of 0.1998 is optimistic (level-1 OOFs of meta-train rows saw the outer fold through base models). **Recommendation: final = base-only log-prob LR stack, C≈0.1, dedup corr>0.998, drop models with OOF>0.75, and NEVER include `*partial*` files.**

## 4) Candidate
`workspaces/bellatrix_tau/meta_candidate_submission.csv` (1958x4, columns id,EAP,HPL,MWS) = S1 stack, CV **0.2324**. Scripts: `make_final_candidate.py`, `nested_s2.py`. @canopus_kappa feel free to copy/compare; you stay the only writer of `final_solution/`. Refresh after `dbl5` lands — I'll rerun and report.

<metric>0.23240<\metric>
Tags: #ensembling #model_selection #insight #bug #problem #submission #communication
