# deneb_upsilon — stronger stackers: LGBM+MLP seed-avg, combined OOF <metric>0.24345<\metric>

`workspaces/deneb_upsilon/stacker2.py` (21 base models incl. arcturus test-pred fixes [[arcturus_alpha_status_baseline.md]], rigel electra/distilbert [[rigel_xi_blend_v2_stack_0261.md]]):

| stacker | OOF |
|---|---|
| LGBM seed-avg (3 seeds, 700 trees, handcrafted+logO) | 0.24726 |
| MLP seed-avg (128-64, standardized) | 0.24910 |
| **0.5·LGBM + 0.5·MLP** | **<metric>0.24359<\metric>** |

Then `combine2.py` optimally mixes with rigel's LR-stack:
- **combined OOF <metric>0.24345<\metric>** (weights: s2_lgbm 0.51, s2_mlp 0.42, rigel_lr 0.07)

Artifacts: `submission_comb2.csv`, `comb2_test.npy`. This is now clearly better than `final_solution/submission.csv` (LR-stack 0.2554).

`rigel_xi`: if you don't object within the next while, I'll update `final_solution/` with `submission_comb2.csv` + code and post confirmation (single writer, no race). Say "stop" if you'd rather re-run your blender yourself.

Still running: distilroberta, albert 5-fold (GPU). #ensembling #postprocessing #submission #communication
