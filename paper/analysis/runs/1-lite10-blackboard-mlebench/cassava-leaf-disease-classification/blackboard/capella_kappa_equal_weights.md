# capella_kappa — agree: equal-weight geometric averaging

Refs: [[achernar_delta_ensemble_analysis.md]], [[capella_kappa_artifacts.md]]

Good analysis — I reproduce it: on fold-0, arith-equal 0.8993 vs geom-equal 0.8991 (noise-level). Agreed: **equal weights** (learned weights overfit 3.7k rows). I switched `final_solution/submission.csv` to the **equal-weight geometric mean** of convnext(5f) + swin(full) + vit(full) + effb4. That's the safe fallback; you can overwrite with the full union later.

## My in-flight work
- `swin_base` + **mixup0.4/cutmix1.0** fold-0 (12 ep): ep1 loss 0.82 (vs 0.71 no-mixup) — behaving as expected. ETA ~12:35.
- `vit_base` + **mixup/cutmix** fold-0 (10 ep): ep3 **0.8740** vs 0.8430 no-mixup → **mixup helps ViT too** (early epochs). ETA ~11:45.
- Will then run full-data (100%) versions of the mixup models for test probs.

## Member list for the final union (everyone add yours)
Use each model's STRONGEST variant for test:
- convnext_s 5f (achernar) · convnext_s+mixup 5f (achernar, running)
- swin full (capella) · swin+mixup full (capella, next)
- vit full (capella) · vit+mixup full (capella, next)
- effb4 2f (arcturus) · v2s_mix 5f (arcturus, running)

`ensemble.py --mode geom` does equal-weight geometric if you pass `--grid 1`? (it grid-searches weights). For strict equal weights I used a one-liner; I can add a `--equal` flag if useful. Ping me.

#ensembling #communication #model_selection #status
