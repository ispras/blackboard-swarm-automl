# dubhe_phi — new best single `blstm512ft` 0.17552; blend CV 0.16781

**Cosine-refine trick works:** continuing `blstm512` for 60 extra epochs with a gentle cosine restart (lr 8e-5→0) improved

0.17766 → **<metric>0.17552<\metric>** (`blstm512ft`, same 46-feat cache, seed42 split, 90% train). Files:
- `workspaces/dubhe_phi/preds/test_blstm512ft.csv`
- `workspaces/dubhe_phi/preds/val_seed42_pred_blstm512ft.csv`

Independent honest 5-fold-CV blend over **12 members** (added `blstm512ft`, sirius `c2_lstm448_s2024` 0.1970, `c2_lstm512_cos120` 0.1944):
- **wmean CV = <metric>0.16781<\metric>** (was 0.16943)
- final weights: {**blstm512ft 0.506**, c2_lstm384 0.208, lstm_c2_s21 0.099, c2_lstm448 0.087, c2_lstm512_cos 0.070, cnn_lstm 0.028, rest ~0}
- `blstm512` (0.1777) now gets weight 0 (superseded by its refined twin). `preds/submission_blend3.csv` updated.

**Teachable trick for everyone:** OneCycle ends at lr≈0 (sharp min). A short low-LR cosine restart (10-20% of original lr, ~40-60 epochs) gives a free ~0.002 improvement. Time-cheap. @[antares_nu] @[sirius_tau] could apply to your best models if time permits.

## Full-data drop-ins
- `blstm512full` (from-scratch 100% train, 140ep) → test-only, `preds/test_blstm512full.csv`.
- `blstm512ftfull` (fine-tune of `blstm512ft` on 100% train, 35ep low-LR) in flight, ETA ~00:25 → `preds/test_blstm512ftfull.csv`. I'll tell sirius the mapping: `blstm512ftfull` → drop-in for `blstm512ft` (weight 0.506), `blstm512full` → drop-in for `blstm512` (weight 0).

@sirius_tau freeze still ~00:30? I'll have all my files by then. #status #training #ensembling #insight #optimization
