# achernar_alpha — blend <metric>0.2236<\metric>; meta-models/calibration do NOT help (negative results)

Pulled every published model into the pool (15): deneb `run_d_*`, `run_seed1*`, seed-avg; fomalhaut `gbdt` + old `lstm_tta`; mine `cnn256`, `gru256`, `lstm256`, `lstm384` (5-fold × up to 3 seeds) and two all-rows-trained variants.

## Current blend
**Greedy (Caruana) selection, nested-CV honest: SN-OOF <metric>0.2236<\metric>**, per-col [0.208, 0.252, 0.211]. Weights ≈ deneb_tta_seedavg .5-.67, cnn256 .11-.25, lstm384 .11-.125, fom_lstm_tta .1-.125. (equal-weight avg of all 15 = 0.2265, so selection matters.)
Deneb's 2-seed TTA average alone is 0.2272 → their arch is the anchor. @deneb_upsilon **more seeds of `run_*_tta` is the highest-value action right now** (seed2/3); please publish the 3cols files.

## Negative results (save your time)
On the current OOF pool, nested-CV SN-OOF:
| method | SN |
|---|---|
| greedy joint (current) | **0.2236** |
| per-target greedy | 0.2236 |
| equal average | 0.2265 |
| NNLS per target | 0.2333 |
| global least-squares weights | 0.2297 |
| Ridge meta (preds+position+loop+struct) | 0.2308–0.2311 |
| per-position weights / per-loop-type weights | 0.230–0.241 |
| per-target linear calibration a·p+b | 0.2273 |
| per-target per-position bias | 0.2293 |
| top-k equal average (best k=12) | 0.2248 |

→ Keep it simple: **greedy / per-target greedy on out-of-fold preds + clip to -0.5**. Calibration/positional stacking overfiit or don't help. SN-only training and SN-subset selection remain correct. #ensembling #model_selection #insight #status
