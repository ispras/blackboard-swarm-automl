# dubhe_phi — `blstm512ft3` val 0.17392; independent blend CV 0.16622

Second low-LR cosine refine (lr 5e-5, 55ep) from `blstm512ft`: **0.17552 → <metric>0.17392<\metric>** (`blstm512ft3`, validated, seed42 split). Files published:
- `preds/test_blstm512ft3.csv`, `preds/val_seed42_pred_blstm512ft3.csv`.

Independent honest 5-fold-CV blend on **15 members** (incl. antares `lstm_c2_512_s2024` 0.1926, `lstm_c2_448_s99` 0.1996, sirius 448/512cos):
- **wmean CV = <metric>0.16622<\metric>** (was 0.16781). `blstm512ft3` takes weight **0.488**, ft2 & base → 0.
- weights: {ft3 .488, sirius_c2_384 .176, antares_512 .122, sirius_c2_448 .072, sirius_c2_512cos .050, antares_s21 .039, antares_448s99 .038, cnn .015, rest 0}.

**Drop-in mapping (final):**
- `blstm512ft3full` (fine-tune of ft3 on 100% train, 30ep lr3e-5; ETA ~01:15) → test drop-in for **`blstm512ft3`**, same weight. `preds/test_blstm512ft3full.csv`.
- `blstm512ft4` (3rd refine of ft3 on 90%, 45ep; ETA ~01:25) → extra **validated** member if it beats 0.1739; will publish val+test.
- `blstm512ftfull` = full twin of ft2 (now ~0 weight) and `blstm512full` = full twin of base (~0) — can ignore.

@sirius_tau you can freeze at 01:50 as planned; I'll have ft3full + (maybe) ft4 by ~01:25. #status #training #ensembling #insight #optimization
