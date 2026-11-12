# Final solution — Google Brain Ventilator Pressure Prediction (swarm)

> **ACTUAL SUBMITTED FILE (final, written 2026-10-22 04:07 local):** the swarm's 16-model fold-0
> blend (global non-negative weights, all-data test-side twins, 3 independent RC-EMA seeds; produced
> by `blend_ensemble.py`). Fold-0 **cross-fitted inspiratory MAE 0.16020**, all-rows **0.15521**
> (in-sample 0.16001 / 0.15513). Chosen over the phase-split 0.16029 candidate because it is
> equivalent on the inspiratory metric and better on the all-rows hedge.
> `submission.csv`: 603,600 rows, ids identical to `data/sample_submission.csv`, 0 NaN,
> clipped [-2, 65], md5 `530d773644e72db6df1fe0fc36d69ebd`. Backups: canopus 0.16029 file
> `0daebb45...` in `workspaces/canopus_iota/candidate_submission.csv`; castor 0.16033 file
> `192ee70c...` in `workspaces/canopus_iota/submission_castor_0246_backup.csv`.
> The 0.16856 values below are an earlier milestone kept for provenance; the latest section at the
> end describes the long-tail/RC-EMA models. See also `canopus_README.md`.

Metric: MAE on inspiratory phase (`u_out == 0`); we also report all-80-step MAE as a hedge.

## Final blend (fold-0 cross-fitted, breath-grouped)
`submission.csv` = per-step-bin weighted ensemble of the swarm's fold-0 models, with test-side
all-data substitutions where available. Blend selection was done only on fold 0 (shared
`GroupKFold(seed=42)`, fold 0 held out); no test feedback used.

Fold-0 cross-fitted CV of the submitted blend:
- inspiratory MAE **0.16856**
- all-rows MAE **0.16018**

## Components (weights fitted on fold 0)
| model | architecture | fold-0 insp MAE |
|---|---|---|
| castor_bilstm_c2 | 4x384 BiLSTM, 46 features | 0.1758 |
| castor_bilstm_big | 3x384 BiLSTM, 38 features | 0.1817 |
| canopus_lstm384_5f2 | 3x384 BiLSTM, 29 features, 5-fold | 0.1965 |
| canopus_lstm512_c2b | 3x512 BiLSTM, 46 features | 0.1997 |
| canopus_lstm384_v2f | 3x384 BiLSTM, 58 features | 0.2057 |
| castor_cnn_gru_c2 | CNN+BiGRU, 46 features | 0.2106 |
| bellatrix_lstm_phys | 3x384 BiLSTM, 46 features | 0.2155 |

Per-step-bin weights were fit with Caruana greedy ensemble selection (with replacement) on the
inspiratory rows; weights are cross-fitted by breath halves on fold 0 to avoid weight overfitting.
For test, the two strongest fold-0 models are replaced by their **all-data (folds 0-4) twins**
(same configurations, same weights), which use 25% more training data.

## Code
- `prep_features.py` — builds the 46-feature breath tensors (`u_in`, lags/leads, diffs, cumsum,
  cumulative volume, physics terms `u_in*R`, `u_in/C`, `vol*R`, `vol/C`, breath-level stats, R/C
  one-hots), z-scored with train statistics.
- `train_bilstm.py` — BiLSTM/Transformer trainer (grouped fold holdout, weighted L1, OneCycle,
  snapshot/SWA averaging).
- `blend_ensemble.py` — loads fold-0 OOF + test predictions from all swarm models, aligns arbitrary
  test orderings by id, fits global/per-(R,C)/per-step-bin weights with cross-fitting, applies
  all-data substitutions, writes `submission.csv`.
- `stack_lgbm.py` — optional LightGBM stacker over base-model predictions (evaluated as an
  alternative to weighted blending).

Reproduce: `python prep_features.py cache3` then `python train_bilstm.py --model lstm --cache cache3
--fold 0 --epochs 250 --hidden 384 --layers 3`, then `python blend_ensemble.py`.

Validated only on training data (fold-0 holdout).

---

# FINAL refresh (2026-10-22 02:15) — fold-0 CV **0.16033** insp

Later long-latency runs improved the blend beyond the 0.16856 version above. The submitted
`submission.csv` corresponds to this final combination (all blend weights fitted on the fold-0
holdout only, no test labels used; greedy non-negative weights, cross-fitted by breath halves gave
held-half 0.1574 / 0.1635 => ~0.1605).

## Fold-0 model scores and weights
| model | fold-0 insp MAE | weight |
|---|---|---|
| castor `bilstm_e3_220` — 4×384 BiLSTM, 51 RC-EMA features, 220 ep, snap5 | 0.17157 | 0.260 |
| castor `bilstm_c2_250` — 4×384 BiLSTM, 46 f, 250 ep cosine, snap5 | 0.17203 | 0.255 |
| canopus `e3_s456` — RC-EMA seed, 170 ep | 0.17414 | 0.169 |
| castor `bilstm_c2` — 4×384, 46 f, 170 ep | 0.17579 | 0.130 |
| canopus `c2s123` — castor recipe, independent seed | 0.17963 | 0.072 |
| castor `bilstm_big` — 3×384, 38 f, 160 ep | 0.18167 | 0.043 |
| canopus `5f2` — 3×384, 29 f, 5-fold | 0.19648 | 0.061 |
| canopus `512c2b` — 3×512, 46 f | 0.19966 | 0.011 |

## Exact test-side composition (sorted-breath layout, clipped [-2,65], joined by id)
```
0.0430  bilstm_v1_all_test          (all-data twin of bilstm_big)
0.0650  bilstm_c2_all_test          } all-data twins of bilstm_c2
0.0650  bilstm_c2_all2_test         }
0.1275  bilstm_c2_250_test          } 250-ep long-tail pair
0.1275  bilstm_c2_all250_test       }
0.1300  bilstm_e3_220_test          } RC-EMA pair
0.1300  bilstm_e3_all_test          }
0.0508  canopus test_pred_lstm384_5f2      (5-fold average)
0.0102  canopus test_pred_lstm384_5f2_all  (all-data, 1/6 mix)
0.0110  canopus test_pred_lstm512_c2b
0.0720  canopus runs/c2s123_test
0.0845  canopus runs/e3_s456_test       } RC-EMA seed pair (fold-0 + all-data)
0.0845  canopus runs/e3_s456_all_test
```
All arrays are ordered by `(breath_id, id)` (same as `shared/test_id_sorted.npy`); arrays in
`sample_submission` order (e.g. bellatrix's) must be reindexed by id before mixing.

## Code added for the final refresh (castor)
- `castor_prep_features.py` — 46-feature breath tensor builder (v2 recipe).
- `castor_build_ema_features.py` — adds 5 RC-exponential-smoothing features (τ≈R·C/1000 s) → 51 f.
- `castor_train_bilstm.py` — the 4×384 BiLSTM trainer used for the 170/220/250-ep runs, with
  weighted L1 (insp 1.0 / exp 0.4), cosine schedule, snapshot-last-5 averaging, and all-data mode.
- `castor_make_submission.py` — weighted test-array → submission builder (sorted-breath layout).
- `castor_blend_weights.py` — greedy non-negative fold-0 blend with breath-half cross-fitting.
- `castor_final_weights.json` — final weights and fold-0 scores.

Key findings: (1) longer low-LR cosine tails (220–250 ep) buy ~0.004–0.005 MAE per model;
(2) RC-EMA features are marginally better per epoch; (3) truncating to 40 steps does NOT help;
(4) post-hoc bias/calibration does not help; (5) the error tail (worst 5% breaths = 32% of MAE)
is dominated by high-R / low-u_in breaths and is the main remaining headroom.
