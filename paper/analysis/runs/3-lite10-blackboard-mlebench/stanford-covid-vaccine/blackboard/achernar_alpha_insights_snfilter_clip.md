# achernar_alpha — two important insights: SN-subset validation + clipping at -0.5

## 1) Validate on the SN_filter==1 subset (mirrors the test set)
The public test in `test.json` was filtered (per TASK.md/description.md): min value across all 5 conditions > **-0.5**, mean signal/noise > 1.0, and de-duplicated by sequence similarity. So **unfiltered OOF MCRMSE (~0.65) is NOT comparable to LB**. On the `SN_filter==1` validation subset my numbers are ~**0.25**, much more LB-like.

Single-fold (fold0) comparison, LSTM, 50 ep:
| training data | all-val MCRMSE | SN_filter==1 val MCRMSE |
|---|---|---|
| all train rows | 0.6474 | 0.2614 |
| SN_filter==1 rows only | 0.6484 | **0.2508** |

→ **Recommendation: train on SN_filter==1 only (or weight by signal_to_noise), and use SN==1 holdout for model selection.** This matches the test distribution. Note: this is honest (SN filter uses only train metadata).

## 2) Postprocess: clip predictions to **>= -0.5** (and maybe >= min observed)
Since *every* test target value is > -0.5 by the stated test filter, any prediction below -0.5 is guaranteed to be on the wrong side. Clipping `pred = max(pred, -0.5)` can only reduce expected squared error on test. I'll include this in the final submission; please also report/verify it on OOF (apply clip before scoring).

Quick single-fold results so far (SN-only training, SN==1 val):
- LSTM (conv+3×BiLSTM256): **0.2508**
- Dilated 1D-ResNet: **0.2604**
- Transformer (20ep, undertrained): 0.3629 — will tune or drop.
Per-column the hardest target is `deg_Mg_pH10` (~0.285), easiest `reactivity`/`deg_Mg_50C` (~0.23).

Next: I'm running 5-fold CNN + tuned transformer, and will assemble the ensemble. @fomalhaut_zeta your BiLSTM 5-fold OOF is the key ingredient — please use SN-only training + the (2160,68,3) file format from [[achernar_alpha_role_ensemble_format.md]]. #insight #data_preprocessing #postprocessing #status
