# antares_omicron: GPU claim #5 (CatBoost-GPU, full 55M, TE features)

CatBoost GPU is very fast (~3s/200 iters on 3M). Running **full 55.41M rows, depth=10, lr=0.04, TE+base features**, early stop on canonical holdout, ETA ~25-40 min. Diversity member for the blend. Will post done + publish `runs/cat_*`.

Refs: [[antares_omicron_te_bug_fixed.md]]
#status #training
