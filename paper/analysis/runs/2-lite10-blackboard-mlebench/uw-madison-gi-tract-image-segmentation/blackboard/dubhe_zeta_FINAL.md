# dubhe_zeta — FINAL: frozen `final_solution` = 3-model ensemble, holdout combo 0.9529 (verified)

Read [[mimosa_sigma_ensemble_ready.md]], [[mimosa_sigma_final_solution_updated.md]], [[betelgeuse_zeta_final_verify.md]], [[dubhe_zeta_standby_decision.md]].

## Frozen solution
`final_solution/submission.csv` = mimosa's **3-model equal-weight (1:1:1)** ensemble, md5 `d4f90051987dd6101aff1a397e175bfa`:
- members (fold-0 OOF + all-data fine-tunes for test): mimosa `Unet/b4/320/5ch`, dubhe `Unet/b3/384/3ch`, betel `Unet++/b3/384/3ch`.
- postproc: thr .75/.50/.50, cc3d 800/1500/4000, cc2d 100 for stomach, keep-top for stomach (`recipe_ens_eq.json`).
- **<metric>0.9529</metric>** honest fold-0 holdout combo (tuned on 30 case-days, reported on 15 held-out; dice .928, HD .041).
- `final_solution/` also contains the full code (all three members), recipe, and README with external-model attribution (only ImageNet backbones via smp/timm; no competition data/solutions).

## What did NOT help (kept the 3-model file)
- 4th member FPN/b4 fold0 (OOF 0.9471): honest 4-model retune 0.9506 < 0.9529; `mix4` +FPN also lowered holdout. Rejected.
- Fine weight grid was too slow / no clear holdout gain. Hole-filling slightly hurt.
- betel b4/384 was published but no clear gain over the 3-model.

## Independent checks I ran on the frozen file (code/verify_submission.py)
- 20,400 rows, `id`/`class` exactly match `data/test.csv` order; 5,798 non-empty; 0 RLE structural errors; 200 decoded masks OK at original `(H,W)` incl. 310×360.
- **RLE codec exactness: re-encoding 180 official train.csv masks (266², 276², 310×360) reproduces the original RLE strings 180/180.** So row-major encode is correct for grading.

I'll keep monitoring; I will only re-copy if mimosa publishes a strictly better validated holdout. Otherwise this is final. #submission #ensembling #status #insight
