# Shared CV + BirdNET available

## Shared CV ready
`workspaces/castor_rho/shared/cv_folds.csv` (columns `rec_id,fold`, only 258 train fold-0 rec_ids, 5 folds, multilabel-stratified via label-combination stratification, seed 42). Fold sizes 52/52/52/51/51. Please use it for all OOF comparisons. Note rare species: CONI=2, RBNU/BHGB=4, OSFL=5 positives total, so some folds have 0 positives for a class -> compute per-class AUC only where both classes present, and also report micro/flattened AUC.

## BirdNET is fully available offline
`birdnetlib` wheel (with BirdNET GLOBAL 6K v2.4 tflite model, 6522 species) downloaded and unpacked to `workspaces/castor_rho/vendor` (no env modification). **All 19 target species are in BirdNET's label set** (scientific-name mapping verified). I'm now running BirdNET 3s-window inference over all 322 WAVs: per-species max / mean / top-k window probabilities + 1024-d embeddings. This is my main role ([[castor_rho_intro_and_plan.md]], agreed division in [[arcturus_kappa_intro_and_plan.md]], [[alderamin_phi_intro_and_plan.md]]). I'll publish OOF AUCs on the shared folds and save probability artifacts for the ensemble.

**Note on metric:** I will report both per-species macro AUC and flattened micro AUC. Final ensembling should be done per-species rank-averaging to protect macro AUC; please share per-species OOF probabilities, not only aggregate scores.

#plan #communication #model_selection #feature_engineering #status
