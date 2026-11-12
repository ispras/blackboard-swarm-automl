# OpenVaccine (MCRMSE) — swarm solution code

Task: predict per-position degradation rates for 107-nt mRNA sequences.
Scored columns: `reactivity`, `deg_Mg_pH10`, `deg_Mg_50C` (positions 0..67). Metric: **MCRMSE** (lower is better).

**Final 5-fold SN-filtered CV of the submitted blend: 0.20533** (cols 0.1886 / 0.2336 / 0.1938).

## Validation protocol
Canonical `KFold(n_splits=5, shuffle=True, random_state=42)` over the 2160 train rows.
Early stopping / model selection / all reported metrics on `SN_filter==1` rows only (the test set is SN-filtered).

## Model families (trained only on `data/train.json`)
All sequence models use one-hot(sequence, structure, loop_type) = 14 channels; loss on the 3 scored
columns only, sample weight = `signal_to_noise` (unless noted).

1. `train_nn.py` — CNN stem + 2L BiGRU / BiLSTM (hidden 256/512, seeds 42/100).
2. `train_ms.py` — multi-scale Conv1d (k=3/5/9) -> 2L BiGRU-512.
3. **`train_struct2.py` — structure-aware model (best family).**
   3-channel base-pair contact tensor per sequence: (binary contact, pair type AU/GC/GU, pair distance)
   -> 2D CNN (3x3, 32 ch, 2 residual blocks) -> row-pooled partner context -> concat with multi-scale
   1D conv -> 2L BiGRU-512 -> 5 outputs.
   Seed bags (each seed is a full 5-fold model; averaged):
   - `struct_scored` (scored-3 loss, 11 seeds): sn1 CV **0.20952**
   - `struct_all5` (all-5 loss, 11 seeds): sn1 CV **0.20862**
   - `struct_huber` (Huber, 2 seeds): sn1 CV **0.21422**

## Ensembling (`ensemble.py`, `build_final.py`)
1. Per scored column: non-negative least squares (ridge-augmented) over OOF predictions; the reported
   number uses 5-fold weight-CV (fit weights on 4 folds, predict the held-out fold).
2. Similarity-gated kNN correction (`knn_final.py`):
   similarity = 0.7*identity(first 68 nt) + 0.3*identity(full 107 nt);
   kNN = similarity^5 weighted mean of the top-3 train neighbours.
   `w = clip((sim_max - 0.85)/0.15, 0, 1) * 0.25`, `p = (1-w)*blend + w*kNN`, then clip at 0.
   (Split-half validated: gain +0.0028, positive 30/30.)
3. Final compact 16-model registry (`code/models_registry_final.json`). Unscored columns are taken from
   an all-5-trained model (`be_mscnn384b`) so the file is valid even if all five columns are scored.
4. `make_submission.py` writes `id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C` for all 240*107 rows.

## Reproduce
```bash
cd <workspace>/workspaces/mirfak_iota
export DATA_DIR=<...>/data
pyenv shell sis-black
# 1) train base models (example: one struct2 seed)
python train_struct2.py --hidden 512 --epochs 130 --patience 28 --sn-weight --seed 42 --tag struct2_all5_s42
# 2) build seed bags + registry (averages all available struct2 seed files, uses models_final_c.json base)
python build_final.py
# 3) kNN arrays (once) and ensemble
python knn_final.py
python ensemble.py --registry final_solution/code/models_registry_final.json \
  --knn-gate --clip 0 --unscored-ref be_mscnn384b \
  --out final_solution/submission.csv
```

## Key findings
- Early stopping on `SN_filter==1`, never on all rows (noise-dominated).
- Loss only on the scored columns + `signal_to_noise` weighting > all-5 MSE and > train-SN1-only.
- 2D base-pair contact features give the single largest gain (~-0.005 MCRMSE vs 1D tokens).
- Seed-bagging struct2 is very effective: 1 seed 0.220 -> 3 seeds 0.212 -> 10 seeds 0.209.
- Gaussian smoothing along positions hurts; per-column affine calibration is negligible; clipping at 0 is harmless.
- GBDT stacking and extra scalar structure descriptors did not help (rejected by CV).
- A compact (~16 model) NNLS registry matches the full 46-model registry CV and is more weight-stable.
