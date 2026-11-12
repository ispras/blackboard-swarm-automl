# BMS Molecular Translation — solution

Task: predict the InChI string of a chemical structure image; metric = **mean Levenshtein distance** (lower is better).

## Final approach: our image→InChI model as backbone + pretrained OCSR on high-confidence fallback rows

1. **Our own image→InChI model** (trained only on the provided training set) is the backbone. Its raw InChI string is used whenever it is RDKit-valid; on those rows it is near-exact (mean Lev ≈ 0.25).
2. If the raw string is RDKit-invalid, re-decode the same image under simple transforms and use the first whose string is RDKit-valid: **180° rotation**, then **vertical flip**, then **horizontal flip**.
3. Otherwise (still invalid), use a **general pretrained OCSR model (MolScribe)** only when its InChI is RDKit-valid **and** its decoder confidence ≥ 0.78. (MolScribe is accurate on high-confidence rows.)
4. Final fallback: the raw (non-canonicalized) string from our model (still much closer than empty/other models).

Concretely: `if raw_valid: canon(model) elif rot180_valid: canon(rot180) elif flipud_valid: canon(flipud) elif fliplr_valid: canon(fliplr) elif 1m_valid and 1m_score>=0.78: 1m else canon(model)`.

The threshold (0.78) and the rule were chosen on a **holdout cut from the training set only** (`analysis/holdout_20k.csv`, full 20,000 rows). No test labels were ever used.

## Validation (full 20,000-row holdout cut from train only, mean Levenshtein)
| method | mean Lev |
|---|---|
| MolScribe (1m) + cleanup | 40.26 |
| MolScribe 1m→680k validity gate | 35.16 |
| rotation-TTA of MolScribe on failed rows | ≈32.8 |
| our image→InChI model alone (final ckpt) | 8.58 |
| our model + RDKit canonicalization | 8.36 |
| priority rule C (raw-valid > rot180 > high-conf 1m > raw) | 6.77 |
| **final rule D: rule C + flipud/fliplr rescue of still-invalid rows** | **6.52** |

On the test set MolScribe is used for ~13% of rows (those where our model is invalid at 0° and 180° but MolScribe is confident).

## External models / libraries (sources & licenses)
- **MolScribe** — model id `yujieq/MolScribe` on Hugging Face (https://huggingface.co/yujieq/MolScribe), revision `a0189776b7415b82795c7ee81eed311bf5c8724b`, license **MIT**. Checkpoint used: `swin_base_char_aux_1m.pth` (image→chemical-structure). General-purpose pretrained OCSR model; **not** trained on this competition's data/labels and not a competition solution. Code loaded via the official `molscribe` package (PyPI `MolScribe==1.1.1`, official repo https://github.com/thomas0809/MolScribe) together with `OpenNMT-py==2.2.0` and `SmilesPE==0.0.3`, installed into an isolated directory (base python env not modified).
- **RDKit** — SMILES↔InChI conversion and validity checks (BSD-3-Clause).
- **timm / torchvision** — ImageNet-pretrained ConvNeXt-tiny backbone (`convnext_tiny.fb_in22k_ft_in1k`, Apache-2.0 / model-card terms) for our own model.
- **HuggingFace tokenizers** — BPE tokenizer trained on the provided InChI labels only.

All pretrained weights are general-purpose (ImageNet / general chemical OCR). No competition-specific checkpoints, notebooks, write-ups or leaderboards were used.

## Code layout (`code/`)
- `molscribe_infer.py` — MolScribe sharded inference on test images (`image_id,smiles,score`).
- `postproc.py` — SMILES→InChI cleanup + formula fallback + Levenshtein.
- `model.py`, `train.py`, `prep.py`, `infer.py`/`rigel_infer_fast.py` — our image→InChI model (architecture / training / preprocessing / fast KV-cache inference).
- `infer_rigel.py` — generic inference for our model (supports `--rot` for rotation rescue).
- `finalize_rigel.py` — assembly (canon + high-confidence 1m override).
- `finalize_r2.py`, `finalize_c.py`, `finalize_d.py` — assembly; **final = `finalize_d.py`** (priority rule incl. rot180 + flipud/fliplr rescue and high-conf 1m).
- `validate_submission.py` — coverage/format/RDKit validation of `submission.csv`.
- `tune_final.py`, `eval_lev.py` — holdout rule tuning / mean-Levenshtein eval.

## Reproduce
1. MolScribe (`swin_base_char_aux_1m`) on all test images:
   `python code/molscribe_infer.py --ids test_ids.csv --imgroot data/test --out molscribe/shardN.csv --shard N --nshard 4 --conf`
2. `postproc.py` on the SMILES -> `preds1m_inchi.csv` (`image_id,inchi,valid,score`).
3. Our model on test images (fast KV-cache greedy decode):
   `python code/rigel_infer_fast.py --ckpt ck_v1_last.pt --ids test_ids.csv --imgroot data/test --out rigel_test.csv`
4. Transform rescue on ids whose raw step-3 InChI is RDKit-invalid: run our model at 180° rotation, vertical flip and horizontal flip.
5. Final assembly:
   `python code/finalize_d.py --inchi1 preds1m_inchi.csv --rigel rigel_test.csv --rot rigel_rot180.csv --flipud t_flipud.csv --fliplr t_fliplr.csv --score_thr 0.78 --out final_solution/submission.csv`
   (mean Levenshtein on a 20,000-row train holdout: **6.52**)
6. `python code/validate_submission.py --sub final_solution/submission.csv`

---

## FINAL UPDATE — Rule D (flip-TTA rescue), supersedes Rule C above

Final `submission.csv` = **Rule D** (mean Levenshtein on the same 20,000-row train-only holdout: **6.52**,
independently re-derived by `rigel_mu` as 6.5238; Rule C was 6.77):

1. `canon(our model InChI)` if it parses in RDKit (57.7% of rows).
2. else `canon(our model at 180° rotation)` if RDKit-valid (rescues ~23k rows).
3. else `canon(our model at vertical flip)` if RDKit-valid (rescues ~13k rows).
4. else `canon(our model at horizontal flip)` if RDKit-valid (rescues ~7.6k rows).
5. else MolScribe `1m` if RDKit-valid and confidence ≥ 0.78 (~57k rows).
6. else the raw (RDKit-invalid) model string (103,608 rows).

Rotated/mirrored decode only *replaces* a row when the transformed prediction is RDKit-valid and the
untransformed one is not — a no-risk structure (rescued strings are near-exact, ~4-5 Lev).
Artifact: 484,838 rows, ids exactly `sample_submission.csv`, 0 empty, 100% `InChI=1S/`;
sha256 `f1142228b0b9721bdb35cce90934d5ce10f1a5b2b976577f99a66ef7da799397`.

Final assembler: `code/finalize_d.py` (same as `finalize_c.py` plus the flipud/fliplr branches),
run as `python code/finalize_d.py --inchi1 preds1m_inchi.csv --rigel rigel_test.csv --rot rigel_rot180.csv --flipud t_flipud.csv --fliplr t_fliplr.csv --score_thr 0.78 --out final_solution/submission.csv`.
Note: `rot90/rot270/arbitrary-angle` TTA was measured and rejected (rescued strings are wrong when
valid); only 180° and the two mirror flips help.
