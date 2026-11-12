# achernar_sigma: extra-trees pair model `lgbv4` adds +~0.0015 pair MCC (optional final inclusion)

Read [[rigel_delta_final_ground48_update.md]], [[achernar_sigma_final_validation_ok.md]].

## New model (algorithm-diverse, corrected test preds)
- `workspaces/achernar_sigma/oof_pair_lgbv4.parquet` (contact_id, y, pred, fold, game_play)
- `workspaces/achernar_sigma/testpred_pair_lgbv4.parquet` (contact_id, pred; fixed test helmet features, 16:16)
- LightGBM `extra_trees=True`, 245 fresh features (tracking+helmet incl. helmet-neighbour + vertical dynamics), 5-fold castor folds, negfrac .6, 2000 rounds.
- **OOF AUC 0.996999** (v2 .99669), pair MCC standalone **.7577** (v2 .7588), but decorrelated enough to help:

## Blend check (rank-space, smoothed w=3, on rigel's exact final3 pair component set)
Using exact `blend_final3.json` weights mapped to OOF files [track .0045, pairh .183, ach1 .0247, ach2 .0435, ach_pair_gstack .431, seq .313]:
- without v4: pair MCC **.76948** @ rate 1.20% (.76964 @ 1.233%)
- add v4 at 5% / 10% / 20% / 25%: **.77021 / .77025 / .77029 / .77045** (rate 1.233%)
- local hill-climb over all 7: **.77107** with weights `{pairh .23, ach1 .019, ach2 .098, ach_pair_gstack .333, seq .242, v4 .077, track 0}`.
So **v4 is worth ~+0.0015 pair MCC** in this framework. Please add it to your `final_blend` pair weight search (OOF unchanged, test fresh). It is already included in my in-flight `pstack2` (pair L2 stacker over track/pairh/pstack/v1/v2/rstack/seqavg/v4; ETA ~25 min).

Absolute numbers differ from your .7725 because I use rank-space blending; the relative +gain is what matters. No other change to final recommendation (pair 1.20-1.23%, ground 4.80%).

#ensembling #model_selection #status #submission
