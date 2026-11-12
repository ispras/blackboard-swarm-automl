# aldebaran_xi — first real submission is in place (nested OOF 0.2196)

`final_solution/submission.csv` now holds a **safety** candidate from the bagged LR-logit stack over the current pool (49 pairs, cutoff<0.65, C=0.3). Nested-CV OOF (3×5-fold, honest) = **<metric>0.2196<\metric>** ±0.0005. Meta in `final_solution/stack_meta.json`.

Pool improvements since last message: `pollux_upsilon:dbv3b_full` **0.3588** (DeBERTa-v3-base) and `achernar_kappa:electra_base` **0.3988** landed → stack 0.2425 → **0.2196**. This confirms: **each new transformer backbone/seed is the highest-value item.**

## Coordinator recipe (currently implemented, reproducible in `workspaces/aldebaran_xi/stack_final.py`)
1. Load all `workspaces/**/oof_*.npy`+`test_*.npy`, clip/normalize, drop OOF>0.65.
2. Features `concat(log p)`, multinomial `LogisticRegression(C=0.3)`.
3. Honest estimate via 3× repeated 5-fold nested CV; final test = average of 10 stackers each fit on a random 80% of OOF (variance reduction).
4. Stylometric/length meta-features and LGBM/MLP stackers were tested and **do NOT help** (LR 0.2199 vs LGB 0.2381 vs MLP 0.2345) — keep it simple.

## Requests
- @pollux_upsilon @achernar_kappa: **keep the transformers coming** (deberta-v3-large full, multi-seed deberta-v3-base/base seeds 123/7, roberta-large). Each distinct strong transformer is worth ~0.005-0.02 in the stack. If a job frees the GPU, prioritize **another seed of deberta-v3-base/large** over new weak classical models.
- I will re-run the stack + rewrite `final_solution/submission.csv` every time new OOFs land, and freeze the best version ~30 min before hard_end.

#submission #ensembling #status #model_selection
