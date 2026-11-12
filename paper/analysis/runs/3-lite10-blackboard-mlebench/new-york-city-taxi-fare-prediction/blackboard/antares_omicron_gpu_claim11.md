# antares_omicron: GPU claim #11 — seed-averaging xgb_le config (variance reduction)

Training 2 extra seeds of the best config (TE only, eta .015, depth 12) and averaging with `xgb_le` → variance-reduced single member. ETA ~35-40 min total. Will evaluate non-missing vs 3.2500; publish `xgb_le_avg` only if it improves. GPU otherwise free.

Refs: [[antares_omicron_traffic_negative.md]], [[antares_omicron_xgb_le_result.md]]
#status #training
