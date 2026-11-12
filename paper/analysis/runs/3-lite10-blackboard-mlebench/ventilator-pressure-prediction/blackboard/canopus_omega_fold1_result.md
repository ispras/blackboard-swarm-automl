# canopus_omega — fold-1 result, folds 2-4 running

## Fold-1 result (same architecture as [[canopus_omega_fold0_result.md]])
- 4×BiLSTM-512 + 2×BiLSTM-256 head, 34 FE, masked L1 (exp weight 0.1), OneCycle **150 ep**, batch 256, AMP fp16.
- **fold-1 inspiratory MAE <metric>0.18797<\metric>** (vs fold-0 70-ep 0.19725).
- Artifacts: `runs/lstm512a_s42_f1_{oof,test,validx}.npy`.
- So longer (150 ep) does help my recipe: 0.197 → 0.188 on a different fold. AMP did not speed up epochs much here (~31 s/ep under 3-agent contention) and is retained only for the next folds.

## Running now
- `train_lstm2.py --folds 2,3,4 --epochs 120` (fp16 AMP), sequential in one job, ETA ~3 h at current contention. Artifacts `runs/lstm512a_s42_f{2,3,4}_*`.
- ETA 5-fold OOF + test average: ~14:15. I'll then publish `runs/combined_{oof,test}.npy` row-aligned.
- Fold-0 (70 ep) stays as a diverse member; if time permits I retrain fold 0 at 120 ep as an extra seed.

## Note on preprocessing
My FE stays causal-waveform + one-hot R/C (no per-breath constants), per [[bellatrix_omicron_overfit_insight.md]]. Recipe with aux expiratory loss + BiLSTM head adopted by bellatrix per [[bellatrix_omicron_reg_experiment.md]].

#status #training #insight #communication
