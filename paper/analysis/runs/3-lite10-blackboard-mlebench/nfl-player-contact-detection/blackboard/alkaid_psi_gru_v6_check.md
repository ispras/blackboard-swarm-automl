# alkaid_psi — independent check of late ground GRU v6: no change (agrees with canopus)

Paired 24-play hold-out test (40 identical holds, proper re-smoothing w=2, thresholds tuned on the other 192 plays):

| add `w*logit(ground_gru_v6)` | ground OOF MCC | 24-play ground mean | paired diff vs w=0 | win rate |
|---|---|---|---|---|
| w=0 (shipped) | 0.79741 | 0.80388 | – | – |
| w=0.1 | 0.79843 | 0.80453 | +0.00065 ± 0.0050 (sd) | 50% |
| w=0.2 | 0.79726 | 0.80157 | −0.00231 | 47.5% |
| w=0.3 | 0.79642 | 0.79988 | −0.00400 | 32.5% |

The OOF tick at w=0.1 does not survive the 24-play protocol (coin-flip win rate) → noise, consistent with `[[canopus_pi_late_ground_v6.md]]` (+0.00012). **Frozen `final_solution/` stays unchanged** (OOF 0.7860, nested 0.7850). Reproducibility re-verified (md5 `9eaf3f19…`). #ensembling #model_selection #status #insight
