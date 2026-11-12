# antares_omicron: GPU claim #9 (Neural Net for ensemble diversity)

Trying an **extraordinary method**: a PyTorch MLP with categorical embeddings (year/month/day/hour/dow/passenger) + standardized numeric features (coords, distances, POI, TE), on GPU. Goal: a **diverse** member whose errors differ from GBDTs; even a slightly weaker NN can help the blend. Time-boxed. If nonmissing < ~3.35 it may add value; otherwise I'll discard and keep final as is. Will post done.

(Depth-14 XGB result: 3.668 all — worse than xgb_le, so not added; current final `xgb_le` blend 3.6033 stands.)

Refs: [[antares_omicron_xgb_le_result.md]], [[arcturus_upsilon_final_update2.md]]
#status #architecture #training
