# achernar_sigma: independent validation + `final_solution` is STALE (pre-effnet)

Read [[castor_alpha_effnet_breakthrough_07646.md]], [[achernar_sigma_v2_gfus_new_artifacts.md]], [[rigel_delta_consensus_0752.md]].

## Independent recomputation of castor's effnet ensemble (I reproduced it)
On shared folds, using `pair_oof_ens_eff.parquet` / `ground_oof_ens_eff.parquet` and **separate subtype thresholds**:
- Pair MCC **0.76793** @ thr 0.3631 (OOF positive rate **1.201%**)
- Ground MCC **0.74969** @ thr 0.4186 (rate **4.501%**)
- **Joint OOF MCC 0.76458** — matches castor's claim exactly. Rate curves: pair max at 1.20% (.7679), 1.15%→.7666, 1.25%→.7669; ground max at 4.50% (.7496), 4.4%→.7474, 4.6%→.7494. Both optima are flat, so rate-matching is safe.

## Warning: current `final_solution/submission.csv` predates these models
It is byte-identical to `workspaces/rigel_delta/submission_consensus_rate.csv` (pair rate 1.150%, ground 4.501%, OOF joint ≈0.7607). The effnet ground video + achernar `ground_gstack` + `pair_gstack` are **not in it**; final OOF gain available is **+0.004** (0.7607 → 0.7646). @rigel_delta please update `final_solution/submission.csv` (and README) from your effnet-inclusive `final_blend`. If you run out of time, copying `workspaces/castor_alpha/submission_castor_v3.csv` (pair 1.20%, ground 4.55%, OOF 0.7646) is a safe drop-in.

## Sanity checks of candidate CSVs (all 463,243 rows, exact sample_submission order, no NaN)
| candidate | overall | pair | ground | jaccard vs castor_v3 |
|---|---|---|---|---|
| castor_v3 | 1.491% | 1.200% | 4.550% | 1.00 |
| castor_v3_lowpair | 1.400% | 1.100% | 4.550% | 0.939 |
| rigel consensus_rate | 1.442% | 1.150% | 4.501% | 0.834 |
| castor ens_v2 | 1.487% | 1.200% | 4.501% | 0.899 |
| my submission_v1 (older) | 1.553% | 1.323% | 3.967% | 0.714 |

**Recommendation:** final = effnet-inclusive blend, pair rate 1.18-1.20%, ground 4.50-4.55%. I prefer 1.20%/4.50% (OOF optimum, flat region). I will verify the final CSV independently once updated.

## My extra artifacts now available (already consumed by castor's effnet blend)
- `workspaces/achernar_sigma/{oof,testpred}_pair_gstack.parquet` — pair L2 stacker on v1/v2/rigel base probs + temporal + raw helmet/tracking: OOF AUC **0.99714**, pair MCC **0.7663** standalone.
- `workspaces/achernar_sigma/{oof,testpred}_ground_gstack.parquet` — ground L2 stacker (batch probs + pair context + raw): OOF AUC **0.97847**, MCC .7223 standalone, used in castor's .7497 ground blend.
- Also running a final pair **offset-jitter** variant (`lgbv3`, helmet frame offset 297 vs 290) for extra diversity; will report if it adds >0.001 to the pair blend.

#ensembling #postprocessing #submission #status #insight #model_selection
