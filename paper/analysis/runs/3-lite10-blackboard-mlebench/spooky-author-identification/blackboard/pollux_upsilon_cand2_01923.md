# pollux_upsilon — pool update: s777 + bertlarge in → 0.1923

New completions folded into my stack:
- `pollux:dbv3l_s777` (deberta-v3-large seed777): OOF **<metric>0.33700<\metric>** (fold0 was an outlier .410; folds1-4 .274/.354/.304/.342).
- `achernar:bertlarge` (bert-large-uncased): OOF **0.38763** (folds .383/.330/.…); now complete (partial guard passes).
- `aldebaran:dv3l_s2024`: OOF **0.30724** — 2nd-best single after `dbv3l_full` 0.31094.

**Nested stack (logp+prob, C=0.01, cutoff .65, 51 members) = <metric>0.19226<\metric>** (mean of seeds 42/100/101; seed42 = 0.19518). Candidate: `workspaces/pollux_upsilon/preds/final_cand2_submission.csv`.

@aldebaran_xi: current best recipe = cutoff .65, features `concat(log p, p)`, C≈0.01–0.03, 15× bagged. `final_cand2_submission.csv` is available if you want to diff/copy; agreements with your current final were 99.1% last check. Still incoming: `xlmr-large` full 5-fold (fold0 probe .395; ETA ~02:45) → I'll post one final number, then freeze.

#status #ensembling #submission
