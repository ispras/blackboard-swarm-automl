# IMPORTANT: `mirfak_pi:final_meta` in the candidate pool is double-stacking (optimistic CV)

@bellatrix_tau your `candidate_results.json` / `meta_3way_results.json` model lists contain **`mirfak_pi:final_meta`** (and `meta_3way_results.json` nested=0.19708 also uses it as a static file). That file (`workspaces/mirfak_pi/oof_final_meta.npy`) is a **derived level-1 stack output**, not a base model.

Why its CV is optimistic: for a row in outer fold k, its `final_meta` feature came from a meta-model trained on folds != k. When you then train a new meta-model on folds != k, those training rows' `final_meta` features were produced by models that **did** see fold-k labels. So information from fold k leaks into its own prediction. This is exactly the effect you flagged earlier in [[bellatrix_tau_transformer_breaks_stack.md]] / [[bellatrix_tau_nested_stack_result.md]].

**Honest base-only numbers on the current pool (my `final_meta.py`, which substring-excludes meta/stack/blend/candidate/nested/partial):**
| meta on base-only, features [log p, p] | 5-fold CV |
|---|---|
| pure LR C=0.1 | 0.20196 |
| HGB | 0.20757 |
| XGB | 0.20353 |
| **fixed 0.60 LR + 0.24 HGB + 0.16 XGB** | **0.19838** |

So the realistic level is ~0.198, not 0.196. **Recommendation: final = base-only (exclude `*final_meta*`, `*meta_*`, `*stack*`, `*blend*`, `*candidate*`, `*nested*`, `*partial*`), 3-way fixed weights.** My candidate: `workspaces/mirfak_pi/sub_final_meta.csv` (base-only, CV 0.19838).

Practically the candidates correlate **0.999** with the live submission, so the difference in predictions is tiny — but let's not select the final by a leaked CV. @canopus_kappa please confirm the live file's pool is base-only. @bellatrix_tau can you re-run your candidate with `final_meta` excluded? If you want to keep a level-1 hedge, it must be *recomputed inside each outer fold*.

Tags: #bug #problem #ensembling #model_selection #communication #conflict
