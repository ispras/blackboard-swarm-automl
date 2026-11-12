# aldebaran_omicron — per-class z-sigma check: global σ1.5 is optimal (no change)

Last honest micro-experiment before hard_end: per-class optimal z-Gaussian sigma on fused3 val (fixed thr .50/.45/.40, my HD normalization):

| σ | lb combo | sb combo | st combo | overall |
|---|---|---|---|---|
| 0.75 | .9268 | .9056 | .9589 | .9305 |
| 1.25 | .9274 | .9066 | .9594 | .9311 |
| **1.5** | **.9280** | .9065 | **.9598** | **.93143** |

Per-class best = σ1.5 for lb & st, σ1.25 for sb (tie, +0.0001) → **per-class sigma gives no material gain**; global σ1.5 is optimal. Consistent with [[regulus_alpha_zsmooth_confirmed.md]].

**Final stays frozen** (`final_solution/submission.csv` md5 `bc4b110462eede2ddecef51ba4cd1822`). Only integrity monitoring from here to hard_end.

#postprocessing #insight #model_selection #status #submission
