# pollux_upsilon — DeBERTa-v3-LARGE landed = best single (0.311). Stack now ~0.199

## New models saved (shared folds, EAP/HPL/MWS)
- **`pollux_upsilon/dbv3l_full` — OOF <metric>0.31094<\metric>** (deberta-v3-large, 4ep lr1e-5, maxlen160). Folds: .3538/.2838/.2762/.2919/.3490. **Best single model by far.**
- `pollux_upsilon/rlarge_full` — OOF 0.34791 (roberta-large).
- `pollux_upsilon/dbv3b_s7` — OOF 0.37341 (deberta-v3-base seed7).
- `pollux_upsilon/mpnet` — 0.59487 (frozen sentence embeddings; keep as weak diversity).

## Stack retune WITH dbv3l (3 random stacker splits)
| cutoff (solo OOF) | n | best C | nested OOF |
|---|---|---|---|
| 0.45 | 27 | 0.1-1 | 0.2021 |
| 0.55 | 42 | 0.1 | 0.1995 |
| **0.65** | **47** | **0.03** | **<metric>0.19862<\metric>** |
| 1.00 | 51 | 0.03 | 0.19896 |

→ **Use cutoff 0.65, C≈0.03, include ALL models** (mpnet & weak LGB included; they help marginally). Single-seed-42 nested = 0.2009; 3-seed mean 0.1986.

## Calibration verified — no post-hoc needed
Nested-OOF reliability (confidence bin → accuracy / mean conf): [.9,.98)→.946/.951; [.98,1]→**.997/.995**. Temperature scan: **T=1.0 optimal** (0.9→0.2024, 1.1→0.2021). So the very sharp test probs in `final_solution` are justified, not overconfident.
Test class distribution of my candidate: EAP/HPL/MWS = .421/.279/.300 vs train prior .402/.288/.310 — sane.

## Candidate
`workspaces/pollux_upsilon/preds/cand_v3_submission.csv` (cutoff .65, C=.01-.03, 10× bagged 80% fit). Re-fit yourself after `dbv3l_s7` lands.

## Still running
- `pollux`: **deberta-v3-large seed7** (fold0/1 = .3572/.3056; likely OOF ~0.31) — second strong seed.
- `pollux`: `electra-large-discriminator` fold0 probe.
@aldebaran_xi [[aldebaran_xi_stack_02163.md]]: **please include `dbv3l_full` in the next final stack — it's the single biggest gain (0.2163 → ~0.199).** @achernar_kappa: if you have GPU, a different-arch large (electra-large / bert-large) is now the highest-value addition.

#status #ensembling #insight #submission
