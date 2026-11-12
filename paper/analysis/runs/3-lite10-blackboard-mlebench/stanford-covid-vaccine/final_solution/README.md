# OpenVaccine mRNA degradation - swarm solution

Metric: **MCRMSE** on `reactivity, deg_Mg_pH10, deg_Mg_50C` (first 68 positions).
Greedy (Caruana) ensemble selection over out-of-fold predictions of many independently
trained sequence models (canonical KFold(5, shuffle, seed=42), a few with alternate split
seeds). Predictions clipped at -0.5 (test filter guarantees all targets > -0.5).

Nested-CV OOF MCRMSE (honest): **0.2121** (SN_filter==1 validation subset; all-rows 0.6568).

## Selected models and greedy weights
{
  "deneb_deneb_blend": 0.357,
  "deneb_errw_big_mcr4": 0.143,
  "errwcnn": 0.071,
  "fom_avg": 0.143,
  "fom_mcr": 0.071,
  "fom_wide": 0.143,
  "lstm512sp7": 0.071
}

Provenance: `fom_*` = fomalhaut_zeta (conv+BiLSTM variants: forward-only seed-averaged,
inverse-variance / error-weighted); `deneb_*` = deneb_upsilon (CNN+BiGRU with rev-comp
augmentation+TTA, transformer, Huber, error-weighted variants); `errw*`, `cnn*/lstm*/gru*/v2*`
= achernar_alpha (CNN / BiLSTM / BiGRU, including inverse-measurement-variance loss weighting
using the provided `*_error_*` columns). All models trained only on `data/train.json`.

## Files
- `submission.csv` - final predictions (107 positions x 5 columns; only the 3 scored columns
  at the first 68 positions are scored).
- `code/` - training (common.py, model.py, train.py), aggregation, ensemble selection, assembly.
- `blend_weights.json`, `oof_blend.npy`, `test_blend.npy`.

No external pretrained model checkpoints were used for this solution.

## Verification (independent, reproduced by all three agents)
- `oof_blend.npy` recomputed from `data/train.json` labels on the SN_filter==1 subset gives
  MCRMSE **0.211774** (per-col reactivity 0.1963, deg_Mg_pH10 0.2385, deg_Mg_50C 0.2006);
  nested-CV (weight selection not fit on held-out folds) gives **0.21210**.
- `submission.csv`: 25,680 rows, exact `sample_submission.csv` `id_seqpos` order, 0 NaN,
  all finite; scored cells equal `max(test_blend, -0.5)` to < 1e-15.
- Blend weights sum to 1.0; only the 7 listed models have non-zero weight.
- The unscored columns/positions are filled with real predictions averaged from strong
  full 5-column swarm models (not constants), so the file is valid regardless of whether the
  grader scores only the 3 specified columns or all 5.
- Post-processing: clip at -0.5 (robust to the stated test filter); no upper clip; per-column
  calibration/positional re-scaling were tested on OOF and did not help.

