# Billion Word Imputation — swarm solution

**Task**: each test sentence has exactly one interior word removed (never first/last). Predict the
completed sentence. Metric: mean **character-level Levenshtein distance** to the original sentence
(lower is better). `submission.csv` = 302,644 rows, ids 0..302643.

## Final approach
1. **Gap detectors (location)** — three detectors (`roberta-base` and two `roberta-large` encoders)
   with a gap-distribution head, each trained on a disjoint slice of `train_v2.txt` by masking
   exactly one random interior token per sentence (reproduces the test hole); their gap posteriors
   are averaged. For each sentence it outputs `p(gap | sentence)` over the `n-1` interior insertion
   slots; argmax = insertion position, max = confidence.
2. **Missing-word prediction** — at the chosen gap, a 4-way log-linear ensemble of masked-LM word
   distributions (`roberta-large` zero-shot 0.38 ⊕ `roberta-large` fine-tuned on train_v2 0.22 ⊕
   `roberta-base` MLM fine-tuned on train_v2 0.15 ⊕ `albert-xxlarge-v2` 0.25). The inserted string is chosen by **expected
   character Levenshtein** over the top-20 words **plus all their prefixes, suffixes,
   single-character deletions, substrings of the top-5, and common words** (length weight λ=0.05).
   Because the metric is character-level, a short hedge can beat committing to a full word when the
   model is uncertain.
3. **Selective insertion** — insert only when detector gap confidence ≥ τ; otherwise leave the test
   sentence unchanged (a wrong-location insertion costs more than the un-filled hole).
4. n-gram LMs (bigram/trigram, Levenshtein-aware candidate refinement) were built and evaluated but,
   after leak-free recalibration, added ~0 over detector+MLM, so they are not in the final decision.

## Validation (leak-free)
Holdout = last 30,000 `train_v2.txt` sentences, one random interior token removed (deterministic
seed). Baseline (no insertion) = **5.5465**. n-gram counts used for calibration were rebuilt
**excluding** this holdout. Detector/MLM were trained only on train_v2 subsets that exclude it.

- Detector gap-argmax accuracy (30k): **0.848**.
- Full pipeline, exact char-Levenshtein on the 30k holdout:
  - ck11000 + 3-way words, τ=0.55: **2.9715**
  - ck15625 + 3-way words, τ=0.60: **2.9082** (0.937 location precision, 0.597 word accuracy)
- **`submission.csv` = 3-detector mix (base + 2×large) + 3-way word ensemble + Levenshtein-optimal
  string hedging + low-confidence additive gate → holdout 2.7423.** Exact decision file and recipe
  in `DECISION.md`.

## External models (general-purpose pretrained only; no competition-specific checkpoints/labels)
| model | HF id | revision | license | purpose |
|---|---|---|---|---|
| RoBERTa base | `roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | MIT | detector encoder (heads trained from scratch on train_v2) |
| RoBERTa large | `roberta-large` | `722cf37b1afa9454edce342e7895e588b6ff1d59` | MIT | zero-shot word model + base for a fine-tuned word model |
| RoBERTa large (fine-tuned) | — | from `roberta-large` | MIT | MLM fine-tune on `train_v2` (700k sentences, 1 epoch, one-mask recipe) as word ensemble member |
| RoBERTa base MLM (fine-tuned) | — | from `roberta-base` | MIT | MLM fine-tune on `train_v2` (~4k steps) as word ensemble member |
| ALBERT xxlarge v2 | `albert-xxlarge-v2` | `97d3e58863d3a41dc581882f73b34d110b18f1f8` | Apache-2.0 | zero-shot masked-LM word ensemble member (decorrelated diversity) |

All fine-tuning used only `train_v2.txt` (no test labels). Validation only on a holdout cut from
`train_v2.txt`.

## Reproduce
- `code/` — detector training (`train_detector.py`, `infer_gapprobs.py`), word models
  (`train_mlm.py`, `word_gaps.py`, `eval_word_gaps.py`), decision assembly
  (`combine_decisions.py`, `write_submission.py`), evaluation (`lev_eval.py`, `pipeline_eval.py`),
  and n-gram experiments (`build_ngram.py`, `build_trigram.py`, `trigram_model.py`, `run_infer.py`).
- `DECISION.md` — the exact decision file used and its holdout score.
