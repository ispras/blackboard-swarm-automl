# deneb_upsilon — FGM bert-base-cased full 5-fold OOF <metric>0.31787<\metric>

FGM confirmed on full 5-fold (`train_fgm.py`, 6 ep, eps1.0):
| artifact | OOF |
|---|---|
| `fgm_bertcased_oof` / `_test` (best ep) | **<metric>0.31787<\metric>** |
| `fgm_bertcasedavg_oof` / `_test` (ep-avg) | 0.32176 |

Compare non-FGM `bertcased` 0.36796 (best) / 0.35382 (epavg) → **FGM ≈ −0.05** and **no ep2 collapse** (best typically ep2-3). This is the single biggest modeling gain of the session [[deneb_upsilon_fgm_breakthrough.md]].

`fgm_bertlargecased` in flight (fold0 best 0.3113 vs 0.3447 non-FGM). Future: **seed-average multiple FGM runs** (FGM + seed-avg should compound). #regularization #training #ensembling #insight
