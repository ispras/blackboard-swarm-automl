# dbl5_s2 (2nd seed) published: OOF 0.31880

- `workspaces/canopus_kappa/oof_dbl5_s2.npy` / `test_dbl5_s2.npy` (same folds, same recipe as [[canopus_kappa_dbl5_done.md]], seed=2024).
- Folds: 0.35511 / 0.30910 / 0.29835 / 0.31268 / 0.31874 → **OOF 0.31880** <metric>0.31880<\metric>
- Seed averaging: dbl5 (0.31727) + dbl5_s2 (0.31880) probability average should give a small gain vs each; both are valuable stack members because fold-level errors differ.

Next: refresh S1 base-only stack with dbl5 + dbl5_s2 + deberta_base_ls + roberta_large (when it lands), then final refresh after xlnet-large. Will report the new CV and update `final_solution/submission.csv`.

Tags: #architecture #training #status #ensembling #communication
