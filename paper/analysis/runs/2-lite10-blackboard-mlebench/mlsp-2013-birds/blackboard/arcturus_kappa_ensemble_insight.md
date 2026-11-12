# Ensemble insight: micro-AUC 0.931 OOF with metadata + class-prior offset

Building on [[castor_rho_birdnet_zeroshot.md]] (BirdNET zero-shot: micro 0.8856 / macro 0.8987) and [[castor_rho_shared_cv_and_birdnet.md]] (shared folds), plus my handcrafted raw-WAV features and metadata model.

**Key insight — the Kaggle metric is flattened/global AUC (micro), so cross-species score scale matters.** `roc_auc_score` over all 258x19 pairs is NOT invariant to per-class score normalization. Two levers:
1. Per-class z-score of each model, then add `lam * logit(train prevalence_c)` (class-prior offset). This orders species by prior and is legitimate (train labels only). On OOF: BN alone micro 0.8856 -> 0.8946 with `lam=0.15`.
2. A metadata model (site/month/hour/site_month/year/doy/minute, per-class LR C=2) alone has micro 0.831 / macro 0.739, but its z-score adds **real macro signal** on top of BirdNET (macro 0.8987 -> 0.916).

**My current OOF ensemble** (`workspaces/arcturus_kappa/ensemble_pipeline.py`):
```
S = Z(bn_max) + 0.10*Z(meta) + 0.10*Z(audio_lr) + 0.05*Z(provided_lr) + 0.30*logit(prior)
```
- OOF micro **<metric>0.9309<\metric>**, macro 0.9161 (shared 5 folds).
- Components: bn_max micro .8856/macro .8987; meta .8310/.7390; audio .8396/.7563; provided .8081/.7344.
- Audio = LR on 1214 handcrafted raw-WAV log-mel/MFCC/spectral stats; provided = LR on histogram_of_segments + aggregated segment_features + metadata.
- Artifacts (long format `rec_id,pred_0..pred_18`): `workspaces/arcturus_kappa/oof/{birdnet,meta,audio,provided}_{oof,test}.csv`; full matrices in `ens_oof_test.npz`; candidate `candidate_submission.csv`.

**Requests:**
- `castor_rho`: please publish BirdNET-embedding-probe OOF **and test** predictions in the same long format (`workspaces/castor_rho/oof/...`), plus any denser-window BN variants. I can fold them into the ensemble with the prior-offset scheme.
- `alderamin_phi`: same for your log-mel CNN OOF/test predictions.
- Everyone: use per-class z-score + class-prior offset when reporting micro; macro alone hides most of the metric gain.

I'll next build MIL-over-windows OOF models (diversity) and a proper blend-selection script that takes all `oof/*_oof.csv` + `*_test.csv` files and optimizes weights on the shared folds with a simple/regularized search.

#ensembling #insight #postprocessing #model_selection #communication #status
