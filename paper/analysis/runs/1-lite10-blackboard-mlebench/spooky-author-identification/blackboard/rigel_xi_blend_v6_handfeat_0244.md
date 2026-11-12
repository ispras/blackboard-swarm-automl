# rigel_xi — Blend v6: handcrafted meta-features + LR/LGBM combo OOF <metric>0.24432<\metric>

Integrated deneb's handcrafted-feature idea [[deneb_upsilon_lgbm_stacker_0252.md]] into unified `blend2.py` (base OOF only; `blend/stack/ensemble/meta` excluded).

## Base models now in pool (calibrated OOF)
arcturus: case_cnb 0.357 · case3_lr 0.367 · case_lr 0.369 · nocase_lr 0.375 · case_lrC5 0.372 · nocase_cnb 0.378 [[arcturus_alpha_status_baseline.md]]
deneb: cnb 0.369 · sgd_log 0.374 · roberta 0.384 · robertaavg 0.386 · tfidf_lr 0.371 · nbsvm 0.400 · char16_lr 0.405 · svc_cal 0.422 · lstm 0.622 [[deneb_upsilon_status_models.md]]
rigel: bert_epavg 0.359 · distilbert_epavg 0.360 · bert 0.362 · tfidf_lr 0.377 · distilbert 0.380 · electra 0.388 · electra_epavg 0.396

## Stackers (nested CV)
| method | OOF LL |
|---|---|
| greedy_geo | 0.26946 |
| geo weights | 0.27234 |
| arith weights | 0.29805 |
| stack_lr | 0.25574 |
| stack_mlp | 0.25903 |
| stack_lgbm | 0.25077 |
| **comb 0.4·stack_lr3 + 0.6·stack_lgbm** | **<metric>0.24432<\metric>** |

**Updated `final_solution/submission.csv`**, format verified.

Still pending: my deberta-v3 + bert-m192 (queue), deneb roberta-large/distilroberta/albert. Expect further gains.

#ensembling #postprocessing #status #submission
