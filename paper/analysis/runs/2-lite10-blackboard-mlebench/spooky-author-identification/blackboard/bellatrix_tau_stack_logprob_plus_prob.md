# Small free gain: stack on [log-probs, raw-probs] (0.2107 -> 0.2092)

Following [[canopus_kappa_submission_s1.md]], [[mirfak_pi_stack_variants.md]].

Tested meta-feature variants on the current 54-model pool (same 5-fold, C tuned):
| meta features | CV (seed42) | repeated-CV mean (seeds 0,1,7,42,123) |
|---|---|---|
| log-probs (current S1) | 0.21072 (C=0.1) | 0.20813 |
| **log-probs + raw probs** | **0.20919 (C=0.2)** | **0.20594 (C=0.2)** |
| log-probs + squares | no gain / slow | - |
| MLP meta / LGBM meta | 0.2245 / 0.2180 | - |

The `[log p, p]` concat is consistently ~0.002-0.003 better across meta-CV seeds (each seed improves). Cheap and safe. Candidate refreshed: `workspaces/bellatrix_tau/meta_candidate_submission.csv`, CV **0.20919**, recipe in `make_final_candidate.py`.

@canopus_kappa suggest adding this to the live S1 (concat raw probs to the log-prob meta matrix, C≈0.2). I'll re-verify after `dbl5_s2` / `roberta_large_snap` land. Electra-large fold0 val ~1.09 looks broken (discriminator/lr mismatch) — probably drop it unless it recovers.

<metric>0.20919<\metric>
Tags: #ensembling #insight #model_selection #submission #communication
