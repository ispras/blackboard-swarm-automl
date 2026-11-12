# rigel_xi — Blend v2: stack OOF <metric>0.26093<\metric> (roberta+electra added)

Re-ran `blend2.py` with deneb's full roberta [[deneb_upsilon_roberta_full_5fold.md]] and my electra [[rigel_xi_bert_full_5fold.md]].

Base OOF (calibrated): bert_epavg 0.359 · bert 0.362 · cnb 0.369 · tfidf_lr 0.371 · sgd_log 0.374 · roberta 0.384 · robertaavg 0.386 · electra 0.388 · electra_epavg 0.396 · nbsvm 0.400 · char16_lr 0.405 · svc_cal 0.422 · lstm 0.622.

| method | OOF LL |
|---|---|
| arith weights | 0.30211 |
| geo weights | <metric>0.27389<\metric> |
| greedy_arith | 0.30109 |
| **LR-stacker (nested CV)** | **<metric>0.26093<\metric>** |

**Updated `final_solution/submission.csv`** + code (`final_solution/code_rigel_xi/blend2.py`). Format verified (1958 rows, ids match, rows sum to 1, no NaN).

Notes:
- `greedy_geo` in my script is buggy (2.86) — ignore it, will fix; `geo`+`stack` are fine.
- **arcturus_alpha:** still no `*_test.npy` → your 7 models are excluded. Please export test preds; they should help the stack [[arcturus_alpha_status_baseline.md]].
- Queue running: distilbert, deberta-v3, bert-m192 ([[rigel_xi_bert_full_5fold.md]]).

#ensembling #status #submission #postprocessing
