# pollux_upsilon — NEGATIVE result: bert-large-cased does not help; FREEZE recommendation

## Final experiment result
`bertlargec_full` (`bert-large-cased`, 4ep lr1.5e-5 bs16, 5-fold): OOF **<metric>0.37992<\metric>** (folds .405/.365/.392/.382/.355). Complete and valid.

But it does **not** improve the stack. My 5-seed comparison (`concat(log p,p)`, C=.01) and @aldebaran's independent `final_select.py` agree:
| pool | n | 5-seed mean |
|---|---|---|
| **no `s555` + no `bertlargec`** | **56** | **<metric>0.18703<\metric>** (aldebaran's seeds) |
| no `bertlargec` | 57 | 0.18721 |
| no `s555` | 57 | 0.18736 |
| all (incl both) | 58 | 0.18750 |

→ **Adding `bert-large-cased` and/or `dv3l_s555` is neutral-to-slightly-harmful (≈+0.0002–0.0005, within noise).** New architectures that helped were electra-large (cased BERT did not). Also `dapt_dv3b` was quarantined by achernar (neutral/harmful) — correct.

## Recommendation for the frozen final
Keep essentially the **current `final_solution/` (56–57 models, nested ≈0.1861)**: prefer the 56-model variant **without `s555` and without `bertlargec`**, C=0.01–0.02, cutoff .65, `concat(log p,p)`, bagged. Do **not** add `bertlargec`/`dapt`; `s555` optional (neutral). No post-processing (T=1, no clipping — confirmed by [[achernar_kappa_verification_report.md]]).

My candidate cross-checks: `final_cand7/9_submission.csv` (without/with bertlargec) — the without-cased one is marginally better. Agreement with your frozen file was 99.5%.

**I am freezing my side now** (no more runs/writes outside my workspace). @aldebaran_xi please finalize `final_solution/` and ping; @achernar_kappa please do the last integrity check. I will only do a read-only verification of the final file before hard_end.

#insight #model_selection #ensembling #submission #status
