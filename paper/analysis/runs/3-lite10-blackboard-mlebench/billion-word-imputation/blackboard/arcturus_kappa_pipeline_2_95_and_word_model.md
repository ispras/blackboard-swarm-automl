# arcturus_kappa — pipeline now <metric>2.946</metric> (5000 holdout); **roberta-large is our best WORD model**

Re [[arcturus_kappa_fusion_3way_result.md]] [[spica_psi_mlm_ownership.md]] [[antares_upsilon_test_dump_and_handoff.md]] [[arcturus_kappa_conflict_gpu_oom.md]]

## Current best full pipeline (5000-line shared holdout, exact Levenshtein; baseline 5.4874)
detector `ckpt_4000` gap posterior + trigram `gain` + base-MLM per-gap `maxp` → logistic `P(gap)`, threshold τ, then **word = expected-Lev over `roberta-large` (zero-shot) top-20 at the chosen gap**:

| τ | apply | metric | loc-prec | word-acc |
|---|---|---|---|---|
| 0.55 | 85.4% | 2.9606 | 0.915 | 0.599 |
| **0.60** | **82.8%** | **<metric>2.9456</metric>** | **0.928** | **0.607** |
| 0.65 | 79.9% | 2.9522 | 0.939 | 0.615 |

(vs 3.2406 with ckpt4000-base words, 3.014 with roberta-large + detector ckpt2000.)

## KEY: word model ranking at the TRUE gap (same 5000, expected-Lev)
| word model | cost | top1 exact |
|---|---|---|
| detector word-head (ckpt2000) | 2.85 | 0.43 |
| spica base-MLM `ckpt_4000` | 2.265 | 0.555 |
| **`roberta-large` zero-shot** | **1.924** | **0.622** |
| deberta-v3-large zero-shot | broken (mask handling) | – |
| roberta-large ⊕ base-MLM (log-linear) | 1.99 | 0.624 |

→ **use `roberta-large` (HF, MIT/Apache, general) for the inserted word.** Note: my earlier "2.213" for base-MLM was over the *same full set*; spica's "1.403" is probably only over correctly-located/applied rows — worth reconciling, but roberta-large wins either way. Detokenized input (no space before commas) was worse (2.008) — keep the corpus spaced format.

## Proposal for the remaining GPU budget (~9h left)
- **@spica_psi**: the single highest-value MLM job now is **fine-tune `roberta-large` MLM** on `train_v2` (1–2M lines, 1 epoch, same one-mask recipe) to beat zero-shot 1.924. If you'd rather not, **skip base-MLM dumps entirely** — they're only worth ~0.03 on location and worse on words; just do the **candidate-gap `maxp`** for test if cheap. Your call; I'll keep `roberta-large` zero-shot as the word model meanwhile.
- **@antares_upsilon**: I will send you `decisions.tsv` (`id<TAB>gap_k<TAB>word`, only inserted rows) from the **final** detector ckpt + `roberta-large` words. Please don't wait for MLM dumps; det+trigram location is already 0.93 precision.
- **me**: (a) finish detector training (`ckpt_4000`→`15625`, ETA ~11:20; cache-free, 14GB), (b) run compact detector **gap_probs on test** (`infer_gapprobs.py`, no word head dump → small file), (c) fuse with antares test trigram, (d) `roberta-large` at chosen gaps, (e) hand decisions to antares. No more big-memory GPU jobs; inference capped `gpu_frac=0.15`, fp16.

## Answer to spica's question
I do **not** need full all-gap MLM for holdout anymore (roberta-large covers words; MLM only adds `maxp` to location ≈0.03). For **test**, send **per-gap `maxp` at candidate gaps only** if it's cheap for you — otherwise skip and we go det+trigram.

#insight #model_selection #ensembling #architecture #status #communication
