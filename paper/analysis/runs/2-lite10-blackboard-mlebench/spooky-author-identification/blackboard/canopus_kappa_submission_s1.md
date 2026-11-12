# `final_solution/submission.csv` refreshed → base-only S1 stack (CV 0.21066)

Per consensus ([[bellatrix_tau_dbl5_stack_0211.md]], [[mirfak_pi_stack_variants.md]], [[canopus_kappa_stack_verified.md]]):
- Recipe: **S1 = base-only**, substring-filtered (`partial/stack/blend/candidate/nested/meta_`), dedup corr>0.998, drop OOF>0.75, multinomial LR on log-probs, **C=0.1**.
- n=54 models, meta-CV **0.21066**; nested agreement across 3 independent implementations (0.21066 / 0.21096 / 0.21101).
- S2 (base+L1) non-nested 0.184 is rejected per nested checks. S1+S3 family-average (0.2199) not better → S1 only.
- Code/recipe: `workspaces/canopus_kappa/make_final.py`, `final_recipe.json`. This is now the live submission.

Will rerun automatically after `dbl5_s2`, `roberta_large_snap`, `electra_large_snap` land. ETA: dbl5_s2 ~60 min, mirfak models ~40-60 min under 3-way GPU sharing.

<metric>0.21066<\metric>
Tags: #ensembling #submission #status #communication
