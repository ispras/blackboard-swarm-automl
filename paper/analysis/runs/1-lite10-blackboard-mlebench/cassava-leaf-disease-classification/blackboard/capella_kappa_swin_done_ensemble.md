# capella_kappa — Swin fold-0 done (0.8916 TTA) + live 3-model fold-0 ensemble

Refs: [[capella_kappa_round2_slots.md]], [[achernar_delta_convnext_fold0.md]], [[arcturus_kappa_effb4_fold0.md]]

## My Swin result
`swin_base_patch4_window12_384.ms_in22k_ft_in1k` @384, fold-0, 10 ep, AdamW OneCycle:
- best single-crop val **<metric>0.8908<\metric>** (ep10)
- with **multi-TTA h+v flip** (`infer.py`): **<metric>0.8916<\metric>**
- artifacts: `workspaces/capella_kappa/swin_base_384/` and `workspaces/capella_kappa/infer_swin/` (val_probs/test_probs).

## Fold-0 ensemble check (single-fold models, arithmetic mean)
| model | fold-0 val acc |
|---|---|
| convnext_s @384 | 0.8932 |
| swin_base @384 | 0.8916 |
| effb4 @512 | 0.8793 |
| **equal avg (all 3)** | **0.8975** |
| **avg convnext+swin (w=0.5)** | **<metric>0.8983<\metric>** |

So we already get **+0.5%** over the best single. Full 5-fold test-avg + more families should push higher.

## Now in `final_solution/submission.csv`
A **valid candidate = 50/50 convnext_s(f0) + swin(f0) test-prob blend** (val 0.8983) — much better than the majority fallback. Will be replaced by the agreed full ensemble at the end. `final_solution/` also needs the solution code — I'll add a runner before hard_end.

## Status
`vit_base_patch16_384.augreg_in21k_ft_in1k` @384 fold-0 just launched (3rd job). @arcturus_kappa: your `f0_test_probs.npy` isn't saved yet (only val) — please save test probs per fold so we can ensemble. @achernar_delta your f0/f1 val+test are there, thanks.

#status #ensembling #inference #submission #architecture #model_selection
