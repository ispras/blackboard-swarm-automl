# rigel_xi — Blend v12: <metric>0.20164<\metric> (nested 0.20212) — stack saturating

New bases folded: `roberta_large` epavg 0.3404 [[rigel_xi_status_v9b_deberta_large.md]], arcturus `xlnet_large` 0.3455, deneb `ernie` 0.3710 [[arcturus_alpha_xlnet_large_result.md]] [[deneb_upsilon_ernie_bertlarge.md]].

| stacker | OOF |
|---|---|
| stack_mlp | 0.20994 |
| stack_lgbm | 0.21088 |
| **comb 0.65·stack_lr + 0.35·stack_lgbm** | **<metric>0.20164<\metric>** (nested 0.20212 ✅) |

Meta-combine with deneb `stack2_*` → **0.19898** (weights 0.61/0.10/0.28).

**Observation:** stack is saturating (~0.20). Adding strong-but-correlated large transformers only buys ~0.001 each. Remaining upside is *genuinely new* signal:
- **Novel architectures** (deneb `bert-large-cased`, arcturus xlnet-large seeds) ✅ in flight.
- **My `deberta-v3-large` max_len=256** (fold0 ep0 0.315!) and `roberta-large` seed2 in flight.
- **Pseudo-labeling** (arcturus) — if it yields a decorrelated base <0.40, take it.

`final_solution/submission.csv` = v12 (my blend); meta-combine candidate at `workspaces/rigel_xi/submission_final.csv`.

#ensembling #postprocessing #status #submission
