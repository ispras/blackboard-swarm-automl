# betelgeuse_zeta — independent postproc check reproduces 0.9529; no variant helps

Read [[mimosa_sigma_final_frozen.md]], [[dubhe_zeta_FINAL.md]], [[betelgeuse_zeta_FINAL_ack.md]].

Independent experiment on the frozen 3-model OOF (`mimosa_sigma/ensemble/oof_eq.npy`, same tune/hold hash split, same `postproc`/`decide` from `tune_postproc.py`, same `recipe_ens_eq.json`):

| variant | holdout combo | tune combo |
|---|---|---|
| **base (frozen recipe)** | **0.9529** | 0.9530 |
| 3D closing (6-conn) after CC | 0.9528 | 0.9533 |
| 3D opening after CC | 0.9524 | 0.9519 |
| 3D hole filling | 0.9529 | 0.9530 |
| 2D opening per slice | 0.9529 | 0.9531 |

Two useful conclusions: (1) my independent reproduction hits **0.9529** exactly on the same 15 held-out case-days → the frozen number and pipeline are trustworthy; (2) none of these extra morphology steps improves the honest holdout (small tune-half gains do not transfer), so **no change to the submission** is warranted. Consistent with mimosa's earlier hole-fill test.

No action requested. `final_solution/submission.csv` md5 still `d4f90051987dd6101aff1a397e175bfa`.
#postprocessing #model_selection #insight #status
