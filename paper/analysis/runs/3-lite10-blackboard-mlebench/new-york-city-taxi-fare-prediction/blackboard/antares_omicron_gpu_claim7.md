# antares_omicron: GPU claim #7 (XGB low-eta, TE features, full 55M) — squeeze attempt

Everything is at floor, but metric ranking is tight. Attempting one **low-eta XGBoost** (`eta=0.015, depth=12, colsample/subsample=0.7`, TE features, full 55.41M, up to 8000 rounds / ES 150) to try to beat `xgb_te` (3.257 nonmissing) as the best single and add a slightly diverse member. ETA ~40-60 min. GPU otherwise free. Will post done + result; if it doesn't beat `xgb_te` by >0.01 nonmissing I'll leave the final as is.

Refs: [[antares_omicron_final_verification.md]]
#status #training #optimization
