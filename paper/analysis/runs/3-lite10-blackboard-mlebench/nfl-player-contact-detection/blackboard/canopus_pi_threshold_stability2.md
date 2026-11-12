# canopus_pi — threshold QA: full-OOF fixed thresholds are the right choice

40 random 24-play holdouts on the frozen final blend (smoothed pp1/gr2); compare threshold policies:

| policy | mean holdout MCC |
|---|---|
| **full-OOF fixed thresholds (shipped)** | **0.79161 ± 0.00284** |
| running-median of nested thresholds | 0.79075 ± 0.00280 |
| per-holdout oracle threshold (tuned on its 192 plays) | 0.78975 ± 0.00284 |

The full-OOF threshold generalizes *better* than a threshold re-tuned on 192 plays — tuning on fewer plays adds noise. This independently confirms `[[alkaid_psi_final_schedule.md]]`'s fixed-threshold decision (and matches the earlier null/negative hedge result). No change to the frozen artifact.

Refs: `[[canopus_pi_hedge_negative.md]]`, `[[canopus_pi_final_qa.md]]`
#postprocessing #model_selection #insight #status
