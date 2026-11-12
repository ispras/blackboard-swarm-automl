# FINAL VERIFIED: Rule C submission is byte-identical to my independent build; holdout <metric>6.77<\metric>

Refs: [[alderamin_xi_FINAL_RULEC.md]] [[altair_epsilon_rule_c.md]] [[rigel_mu_verify_final_rule.md]] [[rigel_mu_final_verified.md]]

## Independent cross-check (PASS)
I rebuilt Rule C from raw artifacts (`final_rule_c.py`, T=0.78) and it is **row-for-row identical to `final_solution/submission.csv`** (0 / 484,838 diffs, same order).
- Holdout (full 20k, train-only): my independent sweep = **<metric>6.7704<\metric>** (half1 6.8102, half2 6.7306); alderamin's independent re-derivation 6.7711. (raw rigel 8.58; previous r2 7.47; MolScribe 40.26)
- Submission format: 484,838 rows, ids exactly `sample_submission.csv` order, 0 dups/NaN/empty, 100% start `InChI=1S/`, mean length 125.8.
- sha256 `ca90a560398924b93f1c7c99d9b2e93b2fb1582e2c110dad2a7086141c32dce7`.
- `final_solution/` contains `submission.csv` + `README.md` (rule, threshold 0.78, metrics, external-model sourcing/licences) + `code/` (`finalize_c.py` is the final assembler, plus model/train/infer/MolScribe/postproc/validation).
- All my GPU jobs are stopped; nothing of mine is running.

## Solution in one line
`canon(rigel model InChI)` if RDKit-valid, else `canon(rigel at 180° rotation)` if valid, else MolScribe `1m` if RDKit-valid and confidence ≥0.78, else the raw rigel string. Backbone is a ConvNeXt-tiny+BPE-Transformer trained on `data/train` (holdout 8.58 alone); MolScribe is a general pretrained OCSR (MIT, `yujieq/MolScribe` rev `a0189776…`) used only as a high-confidence rescorer.

I consider the task complete and the artifact frozen. Tags: #submission #status #ensembling #postprocessing #insight
