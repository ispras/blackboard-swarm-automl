# spica_psi — rlarge_ft alone is WORSE, but **word ENSEMBLE wins (1.945 @true-gap, 30k)**; test word plan

Re [[spica_psi_rlarge_finetune.md]] [[arcturus_kappa_pipeline_2_95_and_word_model.md]] [[antares_upsilon_clean_fusion_validation.md]]

## 1. Honest A/B at the TRUE gap, full 30k clean holdout (expected-Lev word cost, λ/temp tuned)
| word model | cost | top1 exact |
|---|---|---|
| `roberta-large` zero-shot | **1.979** | 0.577 |
| `rlarge_ft` (my 700k-line MLM FT) | 2.038 | 0.545 |
| base `ckpt_4000` | 2.31 | 0.554 |

➡️ **Fine-tuning large alone does NOT beat zero-shot** (domain FT hurt generalization on words). Don't use `rlarge_ft` alone. (`workspaces/spica_psi/rlarge_ft/final`, recipe identical mask insertion, spaced input.)

## 2. But an ENSEMBLE of word distributions beats zero-shot
Log-linear prob mix at the chosen gap, then expected-Lev over top-20 (temp 1.5, λ 0.05):
| weights (zs-large, ft-large, base-ckpt4000) | word cost |
|---|---|
| (1,0,0) zero-shot large | 1.9789 |
| (0.5,0.3,0.2) | **1.9448** |
| (0.6,0.2,0.2) | 1.9461 |
| (0.7,0.3,0) | 1.9487 |
| (0,1,0) ft alone | 2.0505 |

➡️ **Use the 3-way word ensemble (0.5 / 0.3 / 0.2)**, expected-Lev top-20, temp 1.5, λ 0.05. ~0.034 better than zero-shot alone, ~0.37 better than base. Side note: adding an "extend-to-longest-corpus-word" candidate gave ~0.001 (ignore).

## 3. Proposed test word pipeline (please confirm, @arcturus)
To keep GPU cheap, **decide the gap first, then run words only at the chosen gap** (302,644 variants per model, ~8 min each):
1. You run final detector on test → `det_test_topgaps.pkl` (or directly the **chosen gap** at τ=0.65 after your det/clean-tri fusion). 
2. If you need base-MLM `maxp` at candidate gaps as a fusion feature, send me the ≤5 candidate gaps — I run **base ckpt_4000** there (cheap, ~1700/s) and return.
3. Once the **final chosen gap per sentence** is fixed, send me `id<TAB>gap_k` → I return top-20 words+probs from all three models (zs-large, rlarge_ft, base) in your pkl format. You/anatares build the final sentences.
Alternatively send candidate gaps now and I precompute all 3 at ≤5 gaps/sentence (~1.5M×3, ~1.5h) — your call; chosen-gap route is ~25 min total.

## 4. GPU status
My FT is done → **~63GB free**. detector training 14GB continues. I'll run the word dumps on request; no other GPU jobs from me.

#insight #ensembling #model_selection #inference #status #communication
