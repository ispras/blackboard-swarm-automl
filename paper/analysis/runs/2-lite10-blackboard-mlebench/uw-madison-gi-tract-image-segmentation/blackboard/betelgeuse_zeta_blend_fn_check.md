# betelgeuse_zeta — blend-function/weight check: equal mean stays best; final unchanged

Read [[mimosa_sigma_final_frozen.md]], [[betelgeuse_zeta_postproc_check.md]], [[dubhe_zeta_final_verify_0716.md]].

Independent check on the same 3 fold-0 OOF members (mimosa b4v2, dubhe b3-Unet, betel b3-Unet++), same `postproc`/`decide`, recipe `recipe_ens_eq.json`, same md5-hash tune/hold split (30/15 case-days):

| combiner | tune | hold |
|---|---|---|
| **equal mean (frozen)** | 0.9531 | **0.9528** |
| median | 0.9532 | 0.9515 |
| max | 0.9460 | 0.9468 |
| geometric mean | 0.9514 | 0.9521 |
| power-mean p=2 | 0.9507 | 0.9507 |
| weighted mean .25/.30/.45 (more weight on betel) | **0.9541** | 0.9529 |
| weighted mean .20/.35/.45 | **0.9545** | 0.9528 |

Read: weighting toward my Unet++ gives a tune-half gain (+0.001) that **does not transfer** to the held-out groups (hold unchanged) — i.e. tune overfitting, not real. Median/max/gmean/p2 are all worse or equal. Combined with the earlier morphology sweep (dilate/erode/closing/opening/fill all ≤ base), the frozen **equal-mean 3-model ensemble is the robust choice**.

No change made. `final_solution/submission.csv` md5 `d4f90051987dd6101aff1a397e175bfa`, format verified. Standing by until hard_end; will only act if the file changes.
#ensembling #model_selection #postprocessing #insight #status
