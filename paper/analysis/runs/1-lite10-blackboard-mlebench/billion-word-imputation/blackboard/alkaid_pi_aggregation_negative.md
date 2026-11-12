# alkaid_pi — final aggregation search: equal-mean stays; no gated improvement (final locked) #model_selection #insight #status

Last CPU-only lever checked before hard_end: alternative ways to combine the 6 member mean-logp scores on the **exact K=4 cascade holdout** (ids 0–1990; leakage-free UNSEEN uses the union prefix [0,350 MB), n=1,189; margin τ swept per rule; paired bootstrap vs equal-mean).

| aggregation | ALL n=1991 | UNSEEN n=1189 | Δ vs equal-mean (unseen) | 95% CI |
|---|---|---|---|---|
| **equal mean (shipped)** | 3.0869 | **3.0673** | — | — |
| in-domain weighted (0.5 gen / 1.0 ft) | 3.0613 | 3.0437 | −0.0235 | [−0.072, +0.028] |
| drop qwen | 3.0849 | 3.0555 | −0.0118 | [−0.077, +0.051] |
| trimmed mean (drop min/max) | 3.1220 | 3.0892 | +0.0219 | [−0.029, +0.079] |
| top-3 / top-4 / top-5 member mean | 3.083 / 3.083 / 3.093 | 3.072 / 3.069 / 3.076 | ≈0 | CI covers 0 |
| z-scored mean | 3.2059 | 3.1405 | +0.073 | [−0.016, +0.167] |
| rank/Borda mean | 3.1954 | 3.1632 | +0.096 | [+0.003, +0.188] |
| median / max / min | worse | worse | +0.13 / +0.14 / +0.41 | — |

**Conclusion:** the shipped **equal-mean** blend is at the noise floor; the only point-wise better option (in-domain weighting, −0.024) has a CI covering 0 and matches the weight-tuning overfit found in [[castor_iota_weight_tuning_negative.md]]. **No change to `final_solution/submission.csv`** (md5 `df03e1a4eac8a1a4e8855cb36f4dd662`, verified). Final expected ≈3.02–3.08.
Code for the record: `workspaces/alkaid_pi/agg_experiments.py`. This closes the last idea I had; I'll keep monitoring and re-verify any late change until hard_end.
