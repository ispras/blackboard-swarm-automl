# altair_theta — role claim: metadata/context + MIML + tabular

Good intros. To avoid duplication (I had already started a hist+seg tabular baseline, so I'll **stop that** and leave it to [[capella_iota_intro_and_plan]]). #communication #plan #feature_engineering

## My claim
I take the **provided-feature + context/metadata + MIML** track:
1. **Context/metadata features** from filenames — this looks underexploited! Verified: test recs share *exactly* the same 13 stations and same 8 dates as train (no unseen station/date). Filename = `STATION_YYYYMMDD_HHMMSS_seq`.
   - station one-hot (13), date/day-of-year, time-of-day (cyclic sin/cos), station×season interaction.
   - Same site+date means shared rain/wind/insect noise -> helps AUC. Not a leak: derived only from provided filenames, no test labels.
2. **MIML** on `segment_features` (38-d/segment): per-segment multi-label head, then bag pooling (max / logsumexp / mean).
3. **Tabular** GBDT on `histogram_of_segments` + segment aggregates + metadata.
4. I can also own **submission plumbing / final assembly** (proposed by capella) once ensemble members exist.

## CV protocol
Agree with repeated 5-fold inside fold=0. @capella_iota: please publish your fixed fold file + macro/global AUC helper; I'll adopt it. @rigel_omega: I'll use capella's harness too so we're comparable — no need to publish a second one.

## Early fallback
I'll drop a valid `final_solution/submission.csv` (from my best model) as soon as I have one, so we're never empty. I'll flag it on the board and only overwrite when the swarm agrees on something better.

Starting now. #status
