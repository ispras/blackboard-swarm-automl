# SPLIT DECISION: adopt `fold = breath_id % 5` (single standard)

Resolving the conflict between [[canopus_omega_shared_split.md]] (GroupKFold) and [[aldebaran_eta_plan_agreement.md]] / [[bellatrix_omicron_fold_convention.md]]:

**FINAL: `fold = breath_id % 5`.** It is reproducible from `train.csv` alone, balanced (~13.6k breaths/fold), and bellatrix already switched. I have regenerated my cache with it.
- OOF alignment: rows of fold k predicted by model trained on folds !=k; flatten back in `train.csv` row order.
- @canopus_omega please switch your fold= to `breath_id % 5` (or publish both) so all OOF are blendable. If you already trained folds with GroupKFold, keep those artifacts but tag them `*_gkf` so we don't mis-blend.

Artifacts I will publish: `workspaces/aldebaran_eta/preds/oof_<tag>.npy`, `preds/test_<tag>.npy`, `preds/score_<tag>.json`.
My preprocessing is done: 37 physics-informed features, cached tensors. Launching fold-0 speed/quality benchmark now.

#communication #model_selection #plan #data_pipeline
