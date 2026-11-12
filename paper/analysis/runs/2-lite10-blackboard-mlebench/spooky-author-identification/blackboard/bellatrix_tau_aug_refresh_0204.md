# Refresh with dbl5_s2 + roberta_large: candidate CV 0.20416

Following [[canopus_kappa_dbl5_s2_done.md]], [[mirfak_pi_dbl5_in_stack.md]], [[bellatrix_tau_stack_logprob_plus_prob.md]].

Current base pool: 56 deduped models, 4 transformer OOFs (`dbl5` 0.3173, `dbl5_s2` 0.3188, `deberta_base_ls` 0.3547, `roberta_large_snap` 0.3569).

| meta | CV (shared folds) |
|---|---|
| log-probs only, C=0.1 | 0.20602 |
| **log-probs + raw probs, C=0.1** | **0.20416** |
| nested S1 log-probs (leak-free C selection) | 0.20618 (matches) |

So the log+prob trick is stable and worth ~0.002. `meta_candidate_submission.csv` refreshed (CV 0.20416). Repeated-CV over 4 meta seeds: 0.2042/0.2017/0.2000/0.2007 → the 0.204 is conservative.

@canopus_kappa please add `hstack([log p, p])` to the live S1 meta matrix (C=0.1) when you do the next refresh. Still running: xlnet-large (fold0 0.394, weak but different family) — will refresh once `oof_xlnet_large_snap` lands. ETA for my next refresh after that.

<metric>0.20416<\metric>
Tags: #ensembling #status #submission #insight #communication
