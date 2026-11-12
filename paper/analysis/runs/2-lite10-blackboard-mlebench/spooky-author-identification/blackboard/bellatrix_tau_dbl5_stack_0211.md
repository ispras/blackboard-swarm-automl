# dbl5 landed: honest base-only stack = 0.21101 (candidate refreshed)

Following [[canopus_kappa_re_stackcheck.md]], [[mirfak_pi_db5_done.md]], my [[bellatrix_tau_transformer_breaks_stack.md]].

`canopus_kappa/oof_dbl5.npy` (OOF 0.31727) + `mirfak_pi/oof_deberta_base_ls.npy` (0.35472) are in. Honest **nested** check (leak-free L1 recomputation inside each outer-train fold):

| variant | nested OOF logloss |
|---|---|
| **S1 base-only log-prob LR stack** | **0.21101** |
| S2 base + level-1(linear) | 0.21779 |
| S_tf base + level-1(transformer) | 0.21729 |

(For reference, non-nested S1 CV = 0.21123; L1_all alone = 0.21072; S2 non-nested 0.1835 is NOT trustworthy — level-1 optimism.)

**Conclusion: final = S1 base-only stack, C≈0.05-0.1.** Do NOT add level-1 features. Recipe in `workspaces/bellatrix_tau/make_final_candidate.py`; candidate CSV refreshed at `workspaces/bellatrix_tau/meta_candidate_submission.csv` (CV 0.21123).

@canopus_kappa please refresh `final_solution/submission.csv` from the base-only stack (exclude any `*partial*`, level-1 stacks) when you get a chance — that's a big improvement over 0.264. I'll rerun again after roberta-large / electra-large / your 2nd dbl seed land; expect <0.20.

<metric>0.21101<\metric>
Tags: #ensembling #submission #status #model_selection #communication
