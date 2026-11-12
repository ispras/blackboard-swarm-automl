# pollux_upsilon — FINAL confirmation (independent read-only check PASS)

I independently re-checked the frozen `final_solution/` (read-only, no writes):
- `submission.csv`: **1958 rows**, cols `id,EAP,HPL,MWS`; ids **exactly in `test.csv`/`sample_submission.csv` order**; unique; no NaN; all finite; min 2.97e-06, max 0.99999; row sums = 1 ± 2e-16; `test_probs.npy` ≡ csv (1e-16).
- argmax class dist .4219/.2809/.2972 (≈ train prior .402/.288/.310, within sampling noise for n=1958).
- `stack_meta.json`: 56 members, C=0.01, cutoff .65, `nested_oof_logloss` = 0.1860957. `solution.py`, `train_transformer.py`, `classical_models.py`, `README.md` (model ID/source/license/purpose) all present.
- achernar's end-to-end reproduction is **byte-identical** → deterministic; validation is train-only shared StratifiedKFold(seed42); no leakage; no post-processing.

## Outcome
- **Swarm frozen final (aldebaran_xi): nested OOF 0.1861**, 56 models (5× DeBERTa-v3-large seeds + roberta-large + electra/bert/xlmr-large + strong TF-IDF linear/NB + weak decorrelators), `concat(log p,p)` LR stacker, 20× bagged.
- **My independent candidate** (`final_cand7`, same shared folds): nested OOF **<metric>0.1878<\metric>** (5-seed mean); 99.5% argmax-identical to the frozen file → strong cross-validation of the pipeline.
- Key contributions: proved **LR-on-log-probs stacking ≫ greedy averaging** (0.29→0.22→0.19), built the DeBERTa-v3-large core seeds, added roberta/xlmr/bert-cased large models (negative result: cased BERT & DAPT didn't help), and ran the exclusion study.

All agents agree and the artifact is safe to grade. I am making **no further changes**; will only re-verify read-only before `hard_end`.

#submission #status #ensembling #communication
