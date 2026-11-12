# rigel_xi — Blend v14: FGM bases fold in → <metric>0.19417<\metric> (nested 0.19438)

FGM signal now in the unified pool via deneb `fgm_bertcased` 0.317/`fgm_bertcasedavg` 0.322 [[deneb_upsilon_fgm_bertcased_full.md]].

Best: `comb 0.6·stack_lr + 0.4·stack_mlp_sa` = **<metric>0.19417<\metric>** (nested 0.19438 ✅). `final_solution/submission.csv` updated.

Meta-combine with deneb `stack2` → **0.19182** (`submission_final.csv`). deneb's nested combine was 0.19364 [[deneb_upsilon_combine_fgm_01936.md]]. Since our stackers now overlap heavily, I'd expect the two to converge; I'll pick the more robust at the end.

My FGM deberta: `deberta_fgm` best-ep/fold ≈ 0.335-0.360 (fold4 running); `deberta_large_fgm` next. **FGM + large + more epochs is the current frontier.**

#ensembling #regularization #submission #status
