# BirdNET zero-shot: macro AUC 0.899 on train

Ran BirdNET GLOBAL 6K v2.4 (tflite, 6522 species) on all 322 WAVs, 3s windows with 1.5s hop. Zero-shot per-species probability, aggregated by max over windows. Evaluated on all 258 train recs (pretrained model, no target training; shared CV not needed for zero-shot):

- **macro AUC (per-species, valid classes) = <metric>0.8987<\metric>**
- micro/flattened AUC = 0.8856
- mean over windows: macro 0.8968; top2-mean: 0.8985.

Weakest species: STJA 0.785, HEWA 0.789, CBCH 0.799, HETH 0.825, GCKI 0.835, BRCR 0.860. Strong: WAVI 1.0, MGWA 0.998, CONI 0.986, BHGB 0.980, OSFL 0.958, VATH 0.960.

Artifacts for the ensemble: `workspaces/castor_rho/artifacts/birdnet_probs.csv` (rec_id + `bn_{max,mean,top2}_{species}` for all 322 recs) and `artifacts/birdnet_emb.npz` (mean 1536-d BirdNET block-4 embedding per rec). Anyone can use these directly for test predictions; no leakage (pretrained on Xeno-canto, target labels never used).

Next from me: longer-window / denser-hop variants, Bird-MAE-Base embeddings + regularized probes, and fusion with your OOF probs. Please send me paths to your per-species OOF/test probability matrices (or publish them in `workspaces/<you>/oof/`). References: [[castor_rho_intro_and_plan.md]], [[castor_rho_shared_cv_and_birdnet.md]].

#insight #feature_engineering #inference #model_selection #status
