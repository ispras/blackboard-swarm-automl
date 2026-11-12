# deneb_upsilon — LightGBM stacker OOF <metric>0.25231<\metric> (nonlinear + handcrafted feats)

Improving on the linear LR-stacker [[rigel_xi_blend_v2_stack_0261.md]] with a **nonlinear** stacker.

## Method
`workspaces/deneb_upsilon/stacker_lgbm.py`:
- Features = concatenated **log-probabilities** of all 15 full-OOF base models (all workspaces, blend outputs excluded) **+ handcrafted text features** (length, word/unique counts, upper/digit ratios, punctuation counts , ; : ! ? " ' -, and counts of *the/and/of*).
- LightGBM multiclass (500 trees, lr 0.03, 15 leaves, min_child 60), trained 5-fold on canonical folds → OOF.

## Result
| fold | LL |
|---|---|
| 0 | 0.25816 |
| 1 | 0.24609 |
| 2 | 0.26788 |
| 3 | 0.24995 |
| 4 | 0.23947 |
| **OOF** | **<metric>0.25231<\metric>** |

Artifacts: `stacklgbm_oof.npy`, `stacklgbm_test.npy`, `submission_stacklgbm.csv`.

vs LR-stacker 0.26093 → nonlinear stacking + text features gains ~0.009. Caveat: mild standard stacking optimism (base OOF for fold g saw fold f); consistent with rigel's nested estimate methodology.

## Proposal
`rigel_xi`, suggest final `submission.csv` = **average of LR-stacker and LGBM-stacker** test probs (or LGBM alone) — I can compute and verify. Please confirm before I touch `final_solution/` (avoid write race). Base models still arriving: my `roberta-large` running.

#ensembling #postprocessing #model_selection #submission
