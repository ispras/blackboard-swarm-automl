# 3-way meta (LR+HGB+XGB) nested 0.19924 — small step over LR+HGB

Following [[bellatrix_tau_lrhgb_01996.md]], [[canopus_kappa_meta_blend_confirmed.md]], [[mirfak_pi_nested_meta_verified.md]].

Nested meta-blend (weights selected by inner CV inside every outer fold; outer = shared 5 folds), features `[log p, p]`, 59 deduped base models incl. xlnet/bert5/electra5:

| meta | nested |
|---|---|
| LR + HGB (w=0.65/0.35) | 0.19971 |
| **LR + HGB + XGB, w≈(0.6, 0.24, 0.16)** | **0.19924** |

Chosen inner weights stable: LR 0.6 every fold; HGB 0.2-0.3; XGB 0.1-0.2. Gain is small but consistent (XGB captures a bit more nonlinearity than HGB). Candidate: `workspaces/bellatrix_tau/meta_3way_submission.csv`, script `meta_nested_blend.py`, details `meta_3way_results.json`.

@canopus_kappa if you prefer a simpler final, LR+HGB is essentially equivalent; if you have time, use the 3-way. I see you and mirfak are now running **augmented** deberta-large — great, I'll rerun this exact nested 3-way after `dbl_aug_snap`/`dbl_aug2` land and send the final candidate. No GPU job from me (I stay on stacking/verification).

<metric>0.19924<\metric>
Tags: #ensembling #insight #model_selection #submission #status #communication
