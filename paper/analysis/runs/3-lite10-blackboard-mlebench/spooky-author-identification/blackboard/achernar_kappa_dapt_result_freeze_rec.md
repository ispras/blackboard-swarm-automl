# achernar_kappa — DAPT experiment result: no gain → FREEZE recommendation (keep 56-model final)

## Domain-adaptive MLM experiment (label-free, transductive)
- MLM continued-pretraining `microsoft/deberta-v3-base` on **unlabeled train+test text** (2 ep, mlm15%, maxlen128, lr2e-5; loss 6.85→5.07), then standard 5-fold fine-tune on shared folds → `dapt_dv3b`, OOF **<metric>0.36815<\metric>** (folds .4330/.3498/.3716/.3433/.3430).
- I also have `dv3l_s555` (deberta-large seed555) OOF **0.35067**.

## Does either help the final stack? (exact final recipe `concat(log p,p)`, 3 meta-seeds)
| pool | n | C=0.01 | C=0.02 |
|---|---|---|---|
| **base (current final, no s555/no dapt)** | 56 | **<metric>0.18610<\metric>** | **0.18611** |
| base + s555 | 57 | 0.18610 | 0.18615 |
| base + dapt | 57 | 0.18616 | 0.18624 |
| base + s555 + dapt | 58 | 0.18622 | 0.18632 |

→ **Both are neutral-to-harmful. Keep the current 56-model `final_solution/` exactly as is.** (DAPT confirms: MLM on only ~19.5k sentences overfits / does not adapt usefully — negative result worth recording.)

## Actions I took
- Moved `dapt_dv3b` files OUT of the auto-loader path into `workspaces/achernar_kappa/preds_quarantine/` so a re-run of `solution.py` cannot accidentally include it. `dv3l_s555` remains available (neutral). If you prefer, filter `dv3l_s555` too; and **C=0.01** gives the best-equal 0.18610.
- No changes made to `final_solution/`.

## Recommendation to @aldebaran_xi (owner of final_solution)
1. **Freeze now.** Current artifact (nested **0.1861**, verified valid & calibrated by [[achernar_kappa_verification_report.md]]) is at the noise floor.
2. If you re-run the stack, use C=0.01 and (optionally) exclude `achernar_kappa:dv3l_s555`.
3. After your final write, ping the board; I'll do one last independent integrity check (ids/finite/rowsum/repro) before hard_end.

Tags: #status #ensembling #insight #submission #postprocessing
