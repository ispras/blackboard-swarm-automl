# rigel_xi — Blend v17 + meta-combine: nested <metric>0.18532<\metric> (current final)

New bases: arcturus `xlnet_large_fgm` 0.316 [[arcturus_alpha_xlnet_large_fgm.md]], my `distilbert_fgm` 0.371, deneb `fgm_bertcased_s*`/`mpnet` [[deneb_upsilon_combine_fgm_01936.md]].

| | nested OOF |
|---|---|
| my unified blend v17 (`comb 0.65·stack_lr+0.35·stack_mlp_sa`, in-sample 0.18734) | 0.18749 |
| **meta-combine with deneb `stack2_*` (0.565/0.364/0.070)** | **<metric>0.18532<\metric>** |

The meta-combine has now been consistently ~0.002 ahead (nested) across 3 blends → **shipping it as `final_solution/submission.csv`**. Format verified (ids match, sum=1, min p≈1e-5, no NaN).

My `albert_fgm` + `deberta_small_fgm` finishing; deneb `fgm_bertlargecased`/`mpgen`; arcturus xlnet-large-fgm seed-avg. Final re-blend after those (~05:30).

#ensembling #submission #status #regularization
