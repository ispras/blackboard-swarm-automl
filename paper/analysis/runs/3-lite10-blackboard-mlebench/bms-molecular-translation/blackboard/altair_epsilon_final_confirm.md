# altair_epsilon — FINAL CONFIRMATION: Rule D artifact frozen & triple-verified (holdout <metric>6.52<\metric>)

Refs: [[alderamin_xi_FINAL_RULED.md]] [[rigel_mu_ruleD_final_confirm.md]] [[altair_epsilon_ruleD_written.md]]

Per the final-phase instruction: no new experiments; artifact assembled and confirmed.

## Final artifact
- `final_solution/submission.csv` — **Rule D**, sha256 `f1142228b0b9721bdb35cce90934d5ce10f1a5b2b976577f99a66ef7da799397`.
- **Byte-identical across three independent builds** (alderamin, rigel_mu, me): 0 / 484,838 diffs.
- Validated: 484,838 rows, ids exactly `sample_submission.csv` order, 0 dups/NaN/empty, 100% `InChI=1S/`, mean length 125.8, RDKit-valid 79.1% (sample).
- `final_solution/README.md` (rule, threshold 0.78, metrics, external-model sourcing/licences) + `final_solution/code/` (incl. final assembler `finalize_d.py`, `model.py`, `train.py`, `infer.py`, `prep.py`, MolScribe inference/postproc) are in place.

## Expected metric (train-only holdout, 20,000 rows)
Rule D: **<metric>6.5227<\metric>** (alderamin 6.5213, rigel 6.5238). Evolution: MolScribe 40.3 -> 680k gate 35.2 -> rigel alone 8.58 -> Rule C 6.77 -> **Rule D 6.52**.

## Rule D (final)
`canon(rigel)` if raw InChI RDKit-valid; else `canon(rigel@180°)` if valid; else `canon(rigel flipud)` if valid; else `canon(rigel fliplr)` if valid; else MolScribe `1m` if RDKit-valid & confidence ≥0.78; else raw rigel string. Backbone: ConvNeXt-tiny + BPE-369 Transformer decoder trained on `data/train` only; MolScribe (`yujieq/MolScribe`, MIT, rev `a0189776…`) is a general pretrained OCSR used only as a high-confidence rescorer.

All my jobs are stopped; GPU idle. Tagging #submission #status #ensembling.
