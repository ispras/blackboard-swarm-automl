# MLSP 2013 Bird Classification — final solution (swarm ensemble)

Task: multi-label bird-species presence from 10 s audio; metric **ROC-AUC** over all
(recording × species) pairs (flattened/micro AUC). 258 training recordings (fold 0 of
`data/essential_data/CVfolds_2.txt`), 64 test recordings (fold 1). Trained/validated
only on the 258 training labels; test labels are never used.

## Final submission
`submission.csv` — 1216 rows (`Id = rec_id*100 + species`, `Probability` in [0,1]).
File is produced by `solution.py`, which reads the component matrices in `components/`.

## Model
Per-class z-score of each component, weighted sum, plus a class-prior offset:

```
S = 1.00*Z(perch) + 0.15*Z(perch_dn) + 1.00*Z(bn3_dn)
    + 0.20*Z(bn) + 0.10*Z(pp) + 0.10*Z(mlpp) + 0.30*Z(mlpa) + 0.10*Z(meta) + 0.20*Z(pw)
    + 1.10 * [ 0.2*logit(train_prev_c) + 0.8*logit(mean_test_pred_c) ]
where Z(.) = clip(per-class z-score, -1.5, +1.5)
probability = sigmoid(S)          # global monotone map, AUC unchanged
```

| component | description | OOF micro | OOF macro |
|---|---|---|---|
| `perch`  | **Perch v2** zero-shot, 5 s @ 32 kHz windows (2.5 s hop), max over windows | 0.9614 | 0.9753 |
| `perch_dn`| Perch v2 zero-shot on **spectral-gated (denoised)** audio (Wiener-like mask, 15th-percentile noise floor), 5 s @ 32 kHz, max | 0.9579 | 0.9705 |
| `bn`     | **BirdNET** GLOBAL 6K v2.4 zero-shot, 3 s windows / 0.5 s hop, max | 0.8927 | 0.9082 |
| `bn3_dn` | **BirdNET v3.0** zero-shot on **spectral-gated (denoised)** audio (same gate as `perch_dn`), 5 s @ 32 kHz, max over windows | 0.9776 | 0.9842 |
| `pp`     | linear probe on Perch 1536-d embedding | 0.9518 | 0.9003 |
| `mlpp`   | ensemble of 5 MLPs on Perch embedding (PCA-256) + metadata | 0.9669 | 0.9597 |
| `mlpa`   | ensemble of 3 MLPs (512-128-19, dropout/mixup) on Perch emb + BirdNET v3 emb + Perch/BN3 target logits + metadata | 0.9716 | 0.9685 |
| `meta`   | per-class LR on recording metadata (site/month/hour/site_month/year/doy/minute) | 0.8594 | 0.7711 |
| `pw`     | time-weighted same-session label prior from training labels (`exp(-|Δt|/300s)·exp(-|Δseq|/30)`) | 0.7570 | 0.7297 |

**Why the cross-species offset:** the official metric is a single flattened AUC over
1216 rows, so scores of different species are compared with each other. A per-species
constant proportional to `logit(prevalence)` orders species correctly and is worth
~+0.01 AUC. It does not change per-species ranking (macro AUC), only the cross-species
scale. We blend the train-label prevalence with a **transductive** estimate (the mean
predicted probability over the 64 test recordings, from two train-trained MLPs). On
random 64-recording pseudo-test splits this is worth ~+0.003 AUC versus using the train
prior alone; the test split is random (44 site/date groups are mixed train/test), so
the transductive estimate adapts to its class balance. It uses test *inputs* only,
never test labels.

**Final OOF (shared 5-fold, transductive offset):** micro **0.9890**, macro 0.9873.
Honest small-set check (z-score inside each held-out fold of 52 recs, prevalence from
the other folds, transductive): **0.9861** mean (0.9888 / 0.9926 / 0.9868 / 0.9811 / 0.9811),
macro 0.9883. All five folds improve over the previous canonical blend (mean micro
0.9838, macro 0.9861).
Random 64-recording pseudo-test simulations (1000 fresh draws per seed, same draws
for both blends; official split regime): candidate mean micro **0.9869** vs 0.9845
for the previous canonical blend, delta **+0.0023**, win rate 97%, q10 0.9804 vs 0.9780.
Weight/lam sensitivity is flat: clip 1.5-2.0 and lam 1.0-1.2 all reproduce the gain;
the denoised-BN3 replacement is a strictly better view (standalone macro .9842 vs
.9781 for raw BN3), and clipping per-class z-scores prevents a few extreme test
scores from dominating the cross-species scale.

## Files
- `solution.py` — reproduces `submission.csv` from `components/`.
- `submission.csv` — final predictions.
- `components/*_test.csv`, `components/*_oof.csv` — per-component predictions
  (`rec_id,pred_0..pred_18`), OOF on the shared folds.
- `components/cv_folds.csv` — shared 5-fold split used by all swarm members.
- `code/` — component-generation / ensemble code (`common.py`, `mlp_perch.py`,
  `make_final.py`, `bn3_denoise.py`); Perch/BirdNET extraction and probes are by `castor_rho`
  (see board), BMP spectrogram CNN and raw-audio CNN experiments are by
  `alderamin_phi`/`arcturus_kappa`.

## Swarm provenance (blackboard)
Denoised-BN3 + z-clip update (current submission): `blackboard/castor_rho_dnclip_gain.md`,
`blackboard/castor_rho_dnclip_confirmed.md`, `blackboard/castor_rho_dnclip_correction.md`,
`blackboard/alderamin_phi_dnclip_independent_audit.md`.
Perch/BirdNET v2/v3 breakthroughs and embedding probes:
`blackboard/castor_rho_perch_breakthrough.md`, `blackboard/castor_rho_birdnet3_breakthrough.md`,
`blackboard/castor_rho_perch_blend.md`, `blackboard/castor_rho_birdnet_zeroshot.md`,
`blackboard/castor_rho_shared_cv_and_birdnet.md`.
Session/group label prior: `blackboard/castor_rho_prior_and_artifacts.md`,
`blackboard/arcturus_kappa_group_prior_fusion.md`.
Micro-AUC/prior-offset insight and ensemble: `blackboard/arcturus_kappa_ensemble_insight.md`,
`blackboard/arcturus_kappa_perch_blend.md`, `blackboard/arcturus_kappa_transductive_offset.md`. Metadata/MLP components:
`blackboard/alderamin_phi_components_published.md`, `blackboard/alderamin_phi_mlp_and_cnn.md`.
