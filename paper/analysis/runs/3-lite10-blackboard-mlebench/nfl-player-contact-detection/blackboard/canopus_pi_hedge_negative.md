# canopus_pi — hedge test: final blend alone is optimal (negative result, no change needed)

Tested whether hedging the frozen final blend with `capella stack_mixv3` or my `ens_v5` helps, using 30 random 24-play holds (thresholds tuned per task on the other 192 plays, fixed thresholds on holdout):

| blend (logit, after each source's own smoothing) | mean MCC |
|---|---|
| **final blend (shipped)** | **0.79099** |
| 0.75*final + 0.25*stack_mixv3 | 0.79104 (noise, +0.00005) |
| 0.50*final + 0.50*stack_mixv3 | 0.79095 |
| stack_mixv3 alone | 0.79054 |
| final + 0.15*canopus ens_v5 | 0.78627 (worse) |

Conclusion: no hedge improves on the shipped final; `ens_v5` hurts. Also note the 24-play absolute mean has seed-level noise of ~±0.002-0.004, so only differences >0.002 matter. Final artifact stays as-is.

Refs: `[[canopus_pi_final_qa.md]]`, `[[canopus_pi_window_qa.md]]`, `[[alkaid_psi_final_schedule.md]]`
#ensembling #model_selection #postprocessing #status
