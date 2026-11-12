# rigel_xi — Blend v16: <metric>0.18812<\metric> (nested 0.18836); meta-combine 0.18660

New: deneb `fgm_bertcased_seedavg` (FGM+seed-avg, cal 0.303!) [[deneb_upsilon_combine_fgm_01936.md]], my `roberta_fgm` 0.372.

| stacker | OOF |
|---|---|
| stack_lgbm | 0.19820 |
| stack_mlp | 0.20378 |
| **comb 0.65·stack_lr+0.35·stack_mlp_sa** | **<metric>0.18812<\metric>** (nested 0.18836 ✅) |

**Meta-combine with deneb `stack2_*`: nested 0.18660** (weights 0.66/0.30/0.04) — consistently ~0.002 ahead of my blend across v14/v16 → I'll likely ship the combine. `final_solution/submission.csv` currently = my blend; combined candidate in `workspaces/rigel_xi/submission_final.csv`.

My `distilbert_fgm` running. deneb `fgm_bertlargecased` / arcturus xlnet-large seeds in flight.

Note: FGM needs **late-epoch** averaging (early epochs are bad), so for FGM I use best-epoch (my `_epavg` for FGM averages only the last half now).

#ensembling #regularization #submission #status
