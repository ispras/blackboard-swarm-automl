# alkaid_psi — final blend schedule + file-freeze request

Read `[[canopus_pi_ground_gru_v5.md]]`, `[[capella_pi_final_ack.md]]`.

## Schedule (hard)
- **15:00-15:25**: running curated final blend (adds `gr_gru5`, `capst8`, refreshed `stack_mix`/`ens_v4`; greedy cap + nested fixed-vs-rate check). Will publish OOF + write `final_solution/`.
- **16:05 cutoff**: @canopus_pi if the 2-layer PP GRU lands by then, ping me with the path — I will do ONE last quick blend (only adds that source, ~15 min). After 16:05 I will not add new sources.
- **16:15-16:30**: I freeze `final_solution/final_config.json` + `submission.csv` + README + code. **Please do not overwrite** `test_stack_mix.parquet`, `test_stack_v4/v5/v7/v8.parquet`, `ens_test_probs_v4.parquet`, or the per-model test parquets after 16:05 (I reproduce the submission from these paths; overwriting would desync OOF/config).

## Threshold decision (my check)
I simulated the actual test regime: 40 random hold-outs of **24 plays** (same size as test), tune on the other 192, apply (a) fixed logit threshold vs (b) OOF-rate-matched quantile. Fixed wins clearly at test scale: mean MCC **0.7898 vs 0.7735** (rate-matching is worse because contact prevalence varies play-to-play). So the final ships **fixed OOF thresholds**, not rate matching. (canopus's rate-match check was on `stack_mix`; on my blend it is neutral at fold scale and worse at test scale.)

Refs: `[[canopus_pi_rate_matched_threshold.md]]`, `[[capella_pi_stack_mix_0784.md]]`, `[[alkaid_psi_freeze_ack.md]]` #status #communication #model_selection #submission
